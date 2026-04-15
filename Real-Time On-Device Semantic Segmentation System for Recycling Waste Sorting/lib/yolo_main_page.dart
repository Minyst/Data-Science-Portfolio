import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui' as ui;
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;

enum AppPageState { main, guide, camera, result }

class MainPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  const MainPage({super.key, required this.cameras});
  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> with WidgetsBindingObserver {
  AppPageState _currentState = AppPageState.main;

  late CameraController _controller;
  bool _isCameraReady = false;
  bool _hasPermission = false;

  bool _isProcessing = false;
  double _processingProgress = 0.0;
  Timer? _progressTimer;
  String? _processingStatus;

  ui.Image? _overlayImage;
  ui.Image? _predictImage;
  bool _showOverlay = true;
  bool _isCapturing = false;
  
  ui.Image? _originalUiImage;
  double _maskOpacity = 0.75;

  List<List<int>>? _lastLabels;
  List<List<List<double>>>? _processedOutput;

  double _edgeSharpness = 4.0;

  int _imageWidth = 0;
  int _imageHeight = 0;
  List<Map<String, dynamic>> _annotations = [];

  static const MethodChannel _onnx = MethodChannel('onnx');
  List<int>? _inputShape;
  List<int>? _outputShape;
  bool _outputIsNHWC = true;
  bool _isModelLoaded = false;

  static const List<String> _classNames = [
    "background",
    "can",
    "glass",
    "paper",
    "plastic",
    "vinyl",
  ];
  
  static const Map<String, Color> _classColors = {
    "can": Color(0xFF00FFFF),
    "glass": Color(0xFFFFFF00),
    "paper": Color(0xFF80FF00),
    "plastic": Color(0xFFFF0000),
    "vinyl": Color(0xFFFF0080),
  };

  static const int _modelInputSize = 512;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && !_isModelLoaded) {
        _initializeApp();
      }
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _progressTimer?.cancel();
    if (_isCameraReady) { _controller.dispose(); }
    super.dispose();
  }

  Future<void> _initializeApp() async {
    await _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      setState(() { 
        _processingStatus = '모델 로딩 중'; 
        _processingProgress = 0.1; 
      });
      
      final info = await _onnx.invokeMethod('init', { 
        'asset': 'assets/model/model.onnx' 
      });
      
      if (info is Map) {
        _inputShape = (info['inputShape'] as List?)?.map((e) => (e as num).toInt()).toList();
        _outputShape = (info['outputShape'] as List?)?.map((e) => (e as num).toInt()).toList();
        debugPrint('📊 모델 입력 형태: $_inputShape');
        debugPrint('📊 모델 출력 형태: $_outputShape');
      }

      if (_outputShape != null && _outputShape!.length == 4) {
        final n = _outputShape!;
        final numClasses = _classNames.length;
        _outputIsNHWC = (n[3] == numClasses);
        debugPrint('출력 레이아웃: ${_outputIsNHWC ? 'NHWC' : 'NCHW'}');
      }
      
      setState(() { 
        _isModelLoaded = true;
        _processingStatus = '✅ 모델 로딩 완료';
        _processingProgress = 1.0;
      });

      try { 
        await _onnx.invokeMethod('warmup'); 
      } catch (_) {}
      
      await Future.delayed(const Duration(milliseconds: 500));
      setState(() { 
        _processingStatus = null;
        _processingProgress = 0.0;
      });
      
      debugPrint('✅ ONNX 모델 로딩 완료');
    } catch (e) {
      debugPrint('❌ 모델 로딩 실패: $e');
      _showSnackBar('AI 모델 로딩에 실패했습니다: ${e.toString()}');
    }
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status.isGranted) {
      setState(() { _hasPermission = true; });
      await _initializeCamera();
    } else { 
      _showPermissionDialog(); 
    }
  }

  void _showPermissionDialog() {
    showDialog(
      context: context, 
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('카메라 권한 필요'),
        content: const Text('재활용품 분석을 위해 카메라 접근 권한이 필요합니다.'),
        actions: [
          TextButton(
            onPressed: () { 
              Navigator.of(context).pop(); 
              setState(() { _currentState = AppPageState.guide; }); 
            }, 
            child: const Text('취소')
          ),
          TextButton(
            onPressed: () { 
              Navigator.of(context).pop(); 
              openAppSettings(); 
            }, 
            child: const Text('설정으로 이동')
          ),
        ],
      ),
    );
  }

  Future<void> _initializeCamera() async {
    try {
      _controller = CameraController(
        widget.cameras.first,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _controller.initialize();
      try { 
        await _controller.setFlashMode(FlashMode.off); 
      } catch (_) {}
      
      if (mounted) {
        setState(() {
          _isCameraReady = true;
          _currentState = AppPageState.camera;
        });
      }
    } catch (e) {
      _showSnackBar('카메라 초기화에 실패했습니다');
    }
  }

  Float32List _preprocessImage(img.Image image) {
    final resized = img.copyResize(
      image, 
      width: _modelInputSize, 
      height: _modelInputSize,
      interpolation: img.Interpolation.average
    );
    
    final input = Float32List(_modelInputSize * _modelInputSize * 3);
    int index = 0;
    const double rescaleFactor = 0.00392156862745098;  // 1/255
    
    for (int y = 0; y < _modelInputSize; y++) {
      for (int x = 0; x < _modelInputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[index++] = pixel.b * rescaleFactor;  // B
        input[index++] = pixel.g * rescaleFactor;  // G
        input[index++] = pixel.r * rescaleFactor;  // R
      }
    }
    debugPrint('🔧 전처리 완료: rescale_factor=$rescaleFactor (BGR 순서, INTER_AREA-like)');
    return input;
  }

  Future<List<List<List<double>>>> _runInference(Float32List input) async {
    if (_inputShape == null || _outputShape == null) {
      throw Exception('모델이 로드되지 않았습니다');
    }
    final ishape = _inputShape!;
    if (ishape.length != 4) {
      throw Exception('지원하지 않는 입력 텐서 형태: $ishape');
    }
    final bool isNCHW = ishape[1] == 3;
    final int inH = isNCHW ? ishape[2] : ishape[1];
    final int inW = isNCHW ? ishape[3] : ishape[2];
    final bytes = input.buffer.asUint8List();
    
    final res = await _onnx.invokeMethod('run', {
      'input': bytes,
      'width': inW,
      'height': inH,
      'channels': 3,
    }).catchError((e) {
      debugPrint('❌ 네이티브 추론 호출 실패: $e');
      throw Exception('네이티브 추론 실패');
    });
    if (res is! Map) {
      throw Exception('네이티브 추론 실패');
    }
    final data = (res['data'] as List).cast<num>();
    final shape = (res['shape'] as List).cast<num>();
    final H = shape[0].toInt();
    final W = shape[1].toInt();
    final C = shape[2].toInt();
    debugPrint('📊 추론 결과: H=$H, W=$W, C=$C');
    
    final hwc = List.generate(
      H, (_) => List.generate(W, (_) => List.filled(C, 0.0))
    );
    int idx = 0;
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        for (int c = 0; c < C; c++) {
          hwc[y][x][c] = data[idx++].toDouble();
        }
      }
    }
    return hwc;
  }

  List<List<int>> _createSegmentationMask(
    List<List<List<double>>> output, 
    int originalWidth, 
    int originalHeight
  ) {
    final outH = output.length;
    final outW = output[0].length;
    final numClasses = output[0][0].length;

    debugPrint('🔍 모델 출력 형태: H=$outH, W=$outW, Classes=$numClasses');
    
    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;
    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        for (int c = 0; c < numClasses; c++) {
          final val = output[y][x][c];
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
    }
    debugPrint('🔍 출력값 범위: min=${minVal.toStringAsFixed(2)}, max=${maxVal.toStringAsFixed(2)}');
    
    final centerY = outH ~/ 2;
    final centerX = outW ~/ 2;
    debugPrint('🔍 중앙 픽셀($centerX, $centerY) 값:');
    for (int c = 0; c < numClasses; c++) {
      final val = output[centerY][centerX][c];
      debugPrint('  클래스 $c (${c < _classNames.length ? _classNames[c] : "unknown"}): ${val.toStringAsFixed(4)}');
    }
    
    bool needsSoftmax = false;
    final sampleSum = List.generate(numClasses, (c) => output[centerY][centerX][c]).reduce((a, b) => a + b);
    if (sampleSum < 0.9 || sampleSum > 1.1) {
      needsSoftmax = true;
      debugPrint('🔄 Softmax 필요: 합=${sampleSum.toStringAsFixed(4)}');
    } else {
      debugPrint('✅ Softmax 불필요: 합=${sampleSum.toStringAsFixed(4)}');
    }
    
    List<List<List<double>>> processedOutput = output;
    if (needsSoftmax) {
      processedOutput = List.generate(outH, (y) => 
        List.generate(outW, (x) {
          final logits = output[y][x];
          final maxLogit = logits.reduce((a, b) => a > b ? a : b);
          final expSum = logits.map((l) => math.exp(l - maxLogit)).reduce((a, b) => a + b);
          return logits.map((l) => math.exp(l - maxLogit) / expSum).toList();
        })
      );
    }
    _processedOutput = processedOutput;
    
    final avgProbs = List.filled(numClasses, 0.0);
    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        for (int c = 0; c < numClasses; c++) {
          avgProbs[c] += processedOutput[y][x][c];
        }
      }
    }
    debugPrint('🔍 클래스별 평균 확률:');
    for (int c = 0; c < numClasses; c++) {
      avgProbs[c] /= (outH * outW);
      debugPrint('  ${c < _classNames.length ? _classNames[c] : "class$c"}: ${avgProbs[c].toStringAsFixed(4)}');
    }
    
    final labels = _labelsFromProcessed(processedOutput, originalWidth, originalHeight);
    
    final classCounts = List.filled(numClasses, 0);
    for (int y = 0; y < originalHeight; y++) {
      for (int x = 0; x < originalWidth; x++) {
        classCounts[labels[y][x]]++;
      }
    }
    debugPrint('🔍 최종 클래스별 픽셀 개수:');
    for (int c = 0; c < numClasses && c < _classNames.length; c++) {
      final pct = (classCounts[c] * 100.0) / (originalWidth * originalHeight);
      debugPrint('  ${_classNames[c]}: ${classCounts[c]} (${pct.toStringAsFixed(2)}%)');
    }
    return labels;
  }

  Future<void> _captureAndProcess() async {
    if (!_isModelLoaded || !_isCameraReady) {
      _showSnackBar('모델 또는 카메라가 준비되지 않았습니다');
      return;
    }
    if (_isProcessing || _isCapturing) return;

    setState(() { 
      _isCapturing = true; 
      _isProcessing = true; 
      _processingStatus = '촬영 중'; 
      _processingProgress = 0.1; 
    });

    try {
      final shot = await _controller.takePicture();
      final bytes = await shot.readAsBytes();
      final original = img.decodeImage(bytes);
      if (original == null) throw Exception('이미지 디코딩 실패');
      _imageWidth = original.width; _imageHeight = original.height;

      setState(() { _processingStatus = '전처리'; _processingProgress = 0.3; });
      final input = _preprocessImage(original);

      setState(() { _processingStatus = 'AI 분석'; _processingProgress = 0.6; });
      final output = await _runInference(input);

      setState(() { _processingStatus = '결과 생성'; _processingProgress = 0.85; });
      var labels = _createSegmentationMask(output, _imageWidth, _imageHeight);

      // (기존) 경계 노이즈 제거: 다른 클래스도 바뀔 수 있으나 사용자가 OK였으므로 유지
      setState(() { _processingStatus = '경계 정리'; _processingProgress = 0.90; });
      labels = _removeBoundaryNoise(labels, kernelSize: 7, iterations: 3);

      // ⭐ vinyl 구멍 채우기(배경만 보수)
      setState(() { _processingStatus = '구멍 채우기'; _processingProgress = 0.92; });
      labels = _fillVinylHoles_BGOnly(labels);

      // ⭐ vinyl 경계 스무딩(배경만 보수)
      setState(() { _processingStatus = '경계 정리'; _processingProgress = 0.94; });
      labels = _smoothVinylBoundary_BGOnly(labels);

      // ⭐ vinyl 시각화 보강(초경량 브릿지 + 1회 팽창, 배경만)
      setState(() { _processingStatus = '시각화 보강'; _processingProgress = 0.96; });
      labels = _inflateVinylForVisualization(labels, iterations: 1, bridge: true);

      _annotations = _buildSimpleAnnotations(labels);
      _lastLabels = labels;

      final oBytes = Uint8List.fromList(img.encodeJpg(original));
      final oCodec = await ui.instantiateImageCodec(oBytes);
      _originalUiImage = (await oCodec.getNextFrame()).image;

      final maskImg = _labelsToColorMask(labels);
      final mBytes = Uint8List.fromList(img.encodePng(maskImg));
      final mCodec = await ui.instantiateImageCodec(mBytes);
      _predictImage = (await mCodec.getNextFrame()).image;

      setState(() { 
        _processingStatus = '완료'; 
        _processingProgress = 1.0; 
        _currentState = AppPageState.result; 
      });
    } catch (e) {
      _showSnackBar('처리 실패: ${e.toString()}');
    } finally {
      setState(() { 
        _isCapturing = false; 
        _isProcessing = false; 
        _processingStatus = null; 
        _processingProgress = 0.0; 
      });
    }
  }

  img.Image _labelsToColorMask(List<List<int>> labels) {
    final h = labels.length; final w = labels.isEmpty ? 0 : labels[0].length;
    final mask = img.Image(width: w, height: h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final idx = labels[y][x];
        if (idx <= 0 || idx >= _classNames.length) {
          mask.setPixel(x, y, img.ColorRgb8(0,0,0));
        } else {
          final name = _classNames[idx];
          final color = _classColors[name] ?? Colors.white;
          mask.setPixel(x, y, img.ColorRgb8(color.red, color.green, color.blue));
        }
      }
    }
    return mask;
  }

  List<List<int>> _removeBoundaryNoise(List<List<int>> labels, {int kernelSize = 5, int iterations = 2}) {
    final h = labels.length;
    if (h == 0) return labels;
    final w = labels[0].length;
    var current = labels;
    for (int iter = 0; iter < iterations; iter++) {
      final result = List.generate(h, (y) => List.generate(w, (x) => current[y][x]));
      final half = kernelSize ~/ 2;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final centerClass = current[y][x];
          if (centerClass == 0) continue;
          final counts = <int, int>{};
          int totalPixels = 0;
          for (int dy = -half; dy <= half; dy++) {
            for (int dx = -half; dx <= half; dx++) {
              final ny = y + dy;
              final nx = x + dx;
              if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                final cls = current[ny][nx];
                counts[cls] = (counts[cls] ?? 0) + 1;
                totalPixels++;
              }
            }
          }
          int maxCount = 0;
          int majorityClass = centerClass;
          counts.forEach((cls, cnt) {
            if (cnt > maxCount) {
              maxCount = cnt;
              majorityClass = cls;
            }
          });
          if (majorityClass != centerClass && maxCount > totalPixels * 0.5) {
            result[y][x] = majorityClass;
          }
        }
      }
      current = result;
      debugPrint('  ✓ 침범 노이즈 제거 반복 ${iter + 1}/$iterations 완료');
    }
    return current;
  }

  // ⭐ (수정) vinyl 구멍 채우기 - "배경(0)만" vinyl로 메움
  List<List<int>> _fillVinylHoles_BGOnly(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return labels;
    final w = labels[0].length;
    const vinylIdx = 5;
    bool hasVinyl = false;
    for (int y = 0; y < h && !hasVinyl; y++) {
      for (int x = 0; x < w && !hasVinyl; x++) {
        if (labels[y][x] == vinylIdx) hasVinyl = true;
      }
    }
    if (!hasVinyl) return labels;

    var result = List.generate(h, (y) => List.generate(w, (x) => labels[y][x]));
    // 3회 반복, 배경만 변경
    for (int iteration = 0; iteration < 3; iteration++) {
      final nextResult = List.generate(h, (y) => List.generate(w, (x) => result[y][x]));
      bool changed = false;
      for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
          if (result[y][x] != 0) continue; // 배경만 대상
          int vinylNeighbors = 0;
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dy == 0 && dx == 0) continue;
              final ny = y + dy, nx = x + dx;
              if (ny >= 0 && ny < h && nx >= 0 && nx < w && result[ny][nx] == vinylIdx) {
                vinylNeighbors++;
              }
            }
          }
          if (vinylNeighbors >= 5) { // 62.5%
            nextResult[y][x] = vinylIdx;
            changed = true;
          }
        }
      }
      result = nextResult;
      if (!changed) break;
    }

    // 5x5로 한 번 더(배경만)
    final finalResult = List.generate(h, (y) => List.generate(w, (x) => result[y][x]));
    for (int y = 2; y < h - 2; y++) {
      for (int x = 2; x < w - 2; x++) {
        if (result[y][x] != 0) continue; // 배경만
        int vinylCount = 0, totalNeighbors = 0;
        for (int dy = -2; dy <= 2; dy++) {
          for (int dx = -2; dx <= 2; dx++) {
            if (dy == 0 && dx == 0) continue;
            final ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
              totalNeighbors++;
              if (result[ny][nx] == vinylIdx) vinylCount++;
            }
          }
        }
        if (totalNeighbors > 0 && vinylCount >= (totalNeighbors * 0.75).ceil()) {
          finalResult[y][x] = vinylIdx;
        }
      }
    }
    return finalResult;
  }

  // ⭐ (수정) vinyl 경계 스무딩 - "배경(0)만" vinyl로 스냅
  List<List<int>> _smoothVinylBoundary_BGOnly(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return labels;
    final w = labels[0].length;
    const vinylIdx = 5;
    bool hasVinyl = false;
    for (int y = 0; y < h && !hasVinyl; y++) {
      for (int x = 0; x < w && !hasVinyl; x++) {
        if (labels[y][x] == vinylIdx) hasVinyl = true;
      }
    }
    if (!hasVinyl) return labels;

    final result = List.generate(h, (y) => List.generate(w, (x) => labels[y][x]));

    // 5x5에서 vinyl이 많이 둘러싸면 배경을 vinyl로(경계 스냅)
    for (int y = 2; y < h - 2; y++) {
      for (int x = 2; x < w - 2; x++) {
        if (result[y][x] != 0) continue; // 배경만
        int vinylCount = 0;
        for (int dy = -2; dy <= 2; dy++) {
          for (int dx = -2; dx <= 2; dx++) {
            if (dy == 0 && dx == 0) continue;
            final ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w && result[ny][nx] == vinylIdx) {
              vinylCount++;
            }
          }
        }
        if (vinylCount >= 18) { // 약 75%
          result[y][x] = vinylIdx;
        }
      }
    }

    // 3x3 미세 정리(배경만)
    final finalResult = List.generate(h, (y) => List.generate(w, (x) => result[y][x]));
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        if (result[y][x] != 0) continue; // 배경만
        int vinylCount = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue;
            final ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w && result[ny][nx] == vinylIdx) {
              vinylCount++;
            }
          }
        }
        if (vinylCount >= 6) { // 75%
          finalResult[y][x] = vinylIdx;
        }
      }
    }
    return finalResult;
  }

  // ⭐ (신규) vinyl 시각화 보강: 초경량 브릿지 + 1회 팽창 (배경만 변경)
  List<List<int>> _inflateVinylForVisualization(
    List<List<int>> labels, {
      int iterations = 1,
      bool bridge = true,
      int k3x3 = 4, // 3x3 이웃 vinyl >= k 이면 채움(과장 정도)
    }
  ) {
    final h = labels.length;
    if (h == 0) return labels;
    final w = labels[0].length;
    const vinylIdx = 5;

    var current = List.generate(h, (y) => List.generate(w, (x) => labels[y][x]));
    for (int it = 0; it < iterations; it++) {
      final next = List.generate(h, (y) => List.generate(w, (x) => current[y][x]));

      for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
          if (current[y][x] != 0) continue; // 배경만

          // (A) 브릿지: 좌/우 또는 상/하로 vinyl이 맞닿아 있으면 연결
          bool bridged = false;
          if (bridge) {
            if ((current[y][x-1] == vinylIdx && current[y][x+1] == vinylIdx) ||
                (current[y-1][x] == vinylIdx && current[y+1][x] == vinylIdx)) {
              next[y][x] = vinylIdx; 
              bridged = true;
            }
          }
          if (bridged) continue;

          // (B) 3x3 소프트 팽창(배경만)
          int vn = 0;
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dy == 0 && dx == 0) continue;
              if (current[y+dy][x+dx] == vinylIdx) vn++;
            }
          }
          if (vn >= k3x3) {
            next[y][x] = vinylIdx;
          }
        }
      }
      current = next;
    }
    return current;
  }

  List<List<Map<String, dynamic>>> _findConnectedComponents(List<List<int>> labels) {
    final h = labels.length; if (h == 0) return [];
    final w = labels[0].length; final numClasses = _classNames.length;
    final visited = List.generate(h, (_) => List<bool>.filled(w, false));
    final components = <int, List<Map<String, dynamic>>>{};
    for (int c = 1; c < numClasses; c++) {
      components[c] = [];
    }
    void bfs(int startY, int startX, int classId) {
      final queue = <List<int>>[[startY, startX]];
      final points = <List<int>>[];
      visited[startY][startX] = true;
      while (queue.isNotEmpty) {
        final curr = queue.removeAt(0);
        final y = curr[0];
        final x = curr[1];
        points.add([x, y]);
        final dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        for (final dir in dirs) {
          final ny = y + dir[0];
          final nx = x + dir[1];
          if (ny >= 0 && ny < h && nx >= 0 && nx < w && 
              !visited[ny][nx] && labels[ny][nx] == classId) {
            visited[ny][nx] = true;
            queue.add([ny, nx]);
          }
        }
      }
      if (points.length >= (w * h * 0.005)) {
        double sumX = 0, sumY = 0;
        for (final p in points) { sumX += p[0]; sumY += p[1]; }
        final cx = sumX / points.length;
        final cy = sumY / points.length;
        components[classId]!.add({
          'x': cx,
          'y': cy,
          'class': _classNames[classId],
          'confidence': 1.0,
          'size': points.length,
        });
      }
    }
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final classId = labels[y][x];
        if (classId > 0 && classId < numClasses && !visited[y][x]) {
          bfs(y, x, classId);
        }
      }
    }
    return components.values.toList();
  }

  List<Map<String, dynamic>> _buildSimpleAnnotations(List<List<int>> labels) {
    final h = labels.length; if (h == 0) return [];
    final w = labels[0].length;
    final componentsList = _findConnectedComponents(labels);
    final anns = <Map<String, dynamic>>[];
    for (final components in componentsList) {
      anns.addAll(components);
    }
    anns.sort((a, b) => (b['size'] as int).compareTo(a['size'] as int));
    return anns;
  }

  List<List<int>> _labelsFromProcessed(
    List<List<List<double>>> processedOutput,
    int originalWidth,
    int originalHeight,
  ) {
    final outH = processedOutput.length;
    final outW = processedOutput[0].length;
    final numClasses = processedOutput[0][0].length;

    const int vinylIdx = 5;

    final labels = List.generate(
      originalHeight,
      (_) => List<int>.filled(originalWidth, 0),
    );

    for (int y = 0; y < originalHeight; y++) {
      final sy = ((y + 0.5) * outH / originalHeight) - 0.5;
      final y0 = sy.floor().clamp(0, outH - 1);
      final y1 = (y0 + 1).clamp(0, outH - 1);
      final wy1 = sy - y0;
      final wy0 = 1.0 - wy1;

      for (int x = 0; x < originalWidth; x++) {
        final sx = ((x + 0.5) * outW / originalWidth) - 0.5;
        final x0 = sx.floor().clamp(0, outW - 1);
        final x1 = (x0 + 1).clamp(0, outW - 1);
        final wx1 = sx - x0;
        final wx0 = 1.0 - wx1;

        final probs = List<double>.filled(numClasses, 0.0);

        for (int c = 0; c < numClasses; c++) {
          final p00 = processedOutput[y0][x0][c];
          final p10 = processedOutput[y0][x1][c];
          final p01 = processedOutput[y1][x0][c];
          final p11 = processedOutput[y1][x1][c];

          final top = p00 * wx0 + p10 * wx1;
          final bottom = p01 * wx0 + p11 * wx1;
          double p = top * wy0 + bottom * wy1;
          final double sharpness = _edgeSharpness.clamp(2.5, 5.0);
          p = math.pow(p, sharpness).toDouble();
          probs[c] = p.clamp(0.0, 1.0);
        }

        // ⭐ 아주 약하게 vinyl 가중치 보정 (시각화 과장 허용)
        probs[vinylIdx] = (probs[vinylIdx] * 1.06).clamp(0.0, 1.0);

        int maxClass = 0;
        double maxVal = probs[0];
        for (int c = 1; c < numClasses; c++) {
          if (probs[c] > maxVal) {
            maxVal = probs[c];
            maxClass = c;
          }
        }
        labels[y][x] = maxClass;
      }
    }
    return labels;
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message), 
        duration: const Duration(seconds: 3),
        backgroundColor: Colors.black.withValues(alpha: 0.8)
      )
    );
  }

  IconData _statusIcon(String? status) {
    if (status == null) return Icons.pending;
    if (status.contains('촬영')) return Icons.photo_camera;
    if (status.contains('전처리')) return Icons.brush;
    if (status.contains('AI 분석')) return Icons.memory;
    if (status.contains('결과 생성')) return Icons.image;
    if (status.contains('경계 정리')) return Icons.auto_fix_high;
    if (status.contains('완료')) return Icons.check_circle;
    if (status.contains('모델 로딩')) return Icons.download;
    return Icons.pending;
  }

  @override
  Widget build(BuildContext context) {
    switch (_currentState) {
      case AppPageState.main: return _buildMainPage();
      case AppPageState.guide: return _buildMainPage();
      case AppPageState.camera: return _buildCameraPage();
      case AppPageState.result: return _buildResultPage();
    }
  }

  Widget _buildMainPage() {
    return Scaffold(
      backgroundColor: const Color(0xFF4CAF50),
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center, 
            children: [
            Container(
                width: 120, 
                height: 120,
              decoration: BoxDecoration(
                color: Colors.white, 
                borderRadius: BorderRadius.circular(60)
              ),
                child: const Icon(
                  Icons.recycling, 
                  size: 80, 
                  color: Color(0xFF4CAF50)
                )
            ),
            const SizedBox(height: 60),
            
            SizedBox(
              width: MediaQuery.of(context).size.width - 60, 
              height: 70,
              child: ElevatedButton(
                onPressed: () async {
                  if (!_isModelLoaded) {
                    unawaited(_initializeApp());
                  }
                  await _requestCameraPermission();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white, 
                  foregroundColor: const Color(0xFF4CAF50), 
                  elevation: 0, 
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(35)
                    )
                  ),
                  child: const Row(
                    mainAxisAlignment: MainAxisAlignment.center, 
                    children: [
                      Icon(
                        Icons.camera_alt, 
                        size: 28, 
                        color: Color(0xFF4CAF50)
                      ),
                  SizedBox(width: 12),
                      Text(
                        'Take A Photo', 
                        style: TextStyle(
                          fontSize: 24, 
                          fontWeight: FontWeight.w700, 
                          letterSpacing: 1.2, 
                          color: Color(0xFF4CAF50)
                        )
                      ),
                    ]
                  ),
                ),
              ),
            ]
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPage() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: !_hasPermission
          ? const Center(child: CircularProgressIndicator())
          : !_isCameraReady
          ? const Center(child: CircularProgressIndicator())
          : _buildCameraView(),
    );
  }

  Widget _buildCameraView() {
    return Stack(
      children: [
      Positioned(
          top: 0, 
          left: 0, 
          right: 0, 
        height: MediaQuery.of(context).size.height * 0.8, 
        child: ClipRect(child: CameraPreview(_controller))
      ),

      if (_isProcessing || _isCapturing)
        Positioned.fill(
          child: Container(
            color: Colors.black.withValues(alpha: 0.85),
            child: Center(
              child: Container(
                width: 280,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: const Color(0xFF1E1E1E), 
                  borderRadius: BorderRadius.circular(20)
                ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min, 
                    children: [
                      Icon(
                        _statusIcon(_processingStatus), 
                        color: Colors.white, 
                        size: 48
                      ),
                  const SizedBox(height: 12),
                      Text(
                        _processingStatus ?? '처리 중...', 
                        style: const TextStyle(
                          color: Colors.white, 
                          fontSize: 18, 
                          fontWeight: FontWeight.w600
                        )
                      ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: 220,
                    child: LinearProgressIndicator(
                      value: _processingProgress,
                      backgroundColor: Colors.grey[700],
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            Colors.white
                          ),
                    ),
                  ),
                  const SizedBox(height: 8),
                      Text(
                        '${(_processingProgress * 100).round()}%', 
                        style: const TextStyle(
                          color: Colors.white70, 
                          fontSize: 14
                        )
                      ),
                    ]
                  ),
              ),
            ),
          ),
        ),

      Positioned(
          bottom: 0, 
          left: 0, 
          right: 0, 
        height: MediaQuery.of(context).size.height * 0.2,
        child: Container(
          color: Colors.black, 
          child: Center(
          child: GestureDetector(
            onTapDown: (_) => HapticFeedback.lightImpact(),
              onTap: _isProcessing || _isCapturing ? null : _captureAndProcess,
              child: AnimatedContainer(
                  duration: Duration(
                    milliseconds: _isCapturing ? 200 : 100
                  ),
                width: _isCapturing ? 60 : 70, 
                height: _isCapturing ? 60 : 70,
                decoration: BoxDecoration(
                  shape: BoxShape.circle, 
                  color: _isCapturing ? Colors.grey[300] : Colors.white, 
                    border: Border.all(
                      color: Colors.grey[400]!, 
                      width: _isCapturing ? 2 : 3
                    ),
                    boxShadow: _isCapturing 
                      ? [] 
                      : [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.3), 
                            blurRadius: 8, 
                            offset: const Offset(0,2)
                          )
                        ],
              ),
                child: _isCapturing 
                    ? const SizedBox(
                        width: 20, 
                        height: 20, 
                        child: CircularProgressIndicator(
                          strokeWidth: 2, 
                          valueColor: AlwaysStoppedAnimation<Color>(
                            Colors.grey
                          )
                        )
                      ) 
                    : const Icon(
                        Icons.camera_alt, 
                        color: Colors.grey, 
                        size: 30
                      ),
              ),
            ),
          )
        )
      ),
      
      Positioned(
          top: 50, 
          left: 20, 
        child: IconButton(
            onPressed: () { 
              setState(() { 
                _currentState = AppPageState.main; 
              }); 
            }, 
            icon: const Icon(
              Icons.arrow_back_ios, 
              color: Colors.white, 
              size: 24
            )
          )
        ),
      ]
    );
  }

  Widget _buildResultPage() {
    return PopScope(
      canPop: false,
      onPopInvoked: (didPop) async {
        if (!didPop) {
          await _resetToCamera();
        }
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(
      child: Stack(
        children: [
          Positioned.fill(
            child: Center(
              child: _showOverlay
                  ? _buildOverlayWithLabels()
                  : _buildPredictWithLabels(),
            ),
          ),
              
          Positioned(
                top: 20, 
                left: 20,
            child: IconButton(
                  onPressed: () async { 
                    await _resetToCamera(); 
                  },
                  icon: const Icon(
                    Icons.arrow_back_ios, 
                    color: Colors.white, 
                    size: 24
                  ),
            ),
          ),
              
          Positioned(
                bottom: 20, 
                left: 0, 
                right: 0,
            child: Center(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.8), 
                  borderRadius: BorderRadius.circular(25)
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min, 
                  children: [
                    GestureDetector(
                      onTap: () => setState(() => _showOverlay = true),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20, 
                          vertical: 12
                        ),
                        decoration: BoxDecoration(
                          color: _showOverlay 
                            ? Colors.white 
                            : Colors.transparent, 
                          borderRadius: BorderRadius.circular(25)
                        ),
                        child: Text(
                          'Overlay', 
                          style: TextStyle(
                            color: _showOverlay 
                              ? const Color(0xFF4CAF50) 
                              : Colors.white, 
                            fontSize: 16, 
                            fontWeight: FontWeight.w500
                          )
                        ),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => setState(() => _showOverlay = false),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20, 
                          vertical: 12
                        ),
                        decoration: BoxDecoration(
                          color: !_showOverlay 
                            ? Colors.white 
                            : Colors.transparent, 
                          borderRadius: BorderRadius.circular(25)
                        ),
                        child: Text(
                          'Predict', 
                          style: TextStyle(
                            color: !_showOverlay 
                              ? const Color(0xFF4CAF50) 
                              : Colors.white, 
                            fontSize: 16, 
                            fontWeight: FontWeight.w500
                          )
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    ),
  ),
);
  }

  Widget _buildOverlayWithLabels() {
    if (_lastLabels == null || _originalUiImage == null) {
      return const SizedBox.shrink();
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final cw = constraints.maxWidth;
        final ch = constraints.maxHeight;
        final imgW = _imageWidth.toDouble();
        final imgH = _imageHeight.toDouble();

        final scale = math.min(cw / imgW, ch / imgH);
        final dw = imgW * scale;
        final dh = imgH * scale;
        final dx = (cw - dw) / 2.0;
        final dy = (ch - dh) / 2.0;

        final children = <Widget>[
          Positioned(
            left: dx, 
            top: dy, 
            width: dw, 
            height: dh,
            child: RawImage(
              image: _originalUiImage!, 
              fit: BoxFit.contain,
              filterQuality: FilterQuality.none,
              isAntiAlias: false,
            ),
          ),
        ];

        if (_predictImage != null) {
          children.add(
            Positioned(
              left: dx, 
              top: dy, 
              width: dw, 
              height: dh,
            child: Opacity(
              opacity: _maskOpacity,
                child: RawImage(
                  image: _predictImage!, 
                  fit: BoxFit.contain,
                  filterQuality: FilterQuality.none,
                  isAntiAlias: false,
                ),
              ),
            )
          );
        }

        children.add(
          Positioned(
            left: dx, 
            top: dy, 
            width: dw, 
            height: dh,
          child: CustomPaint(
            size: Size(dw, dh),
            painter: _AnnotationsPainter(
              annotations: _annotations,
              xScale: dw / imgW,
              yScale: dh / imgH,
            ),
          ),
          )
        );

        return Stack(children: children);
      },
    );
  }

  Widget _buildPredictWithLabels() {
    if (_predictImage == null) return const SizedBox.shrink();

    return LayoutBuilder(
      builder: (context, constraints) {
        final cw = constraints.maxWidth;
        final ch = constraints.maxHeight;
        final imgW = _imageWidth.toDouble();
        final imgH = _imageHeight.toDouble();

        final scale = math.min(cw / imgW, ch / imgH);
        final dw = imgW * scale;
        final dh = imgH * scale;
        final dx = (cw - dw) / 2.0;
        final dy = (ch - dh) / 2.0;

        return Stack(
          children: [
            Positioned(
              left: dx, 
              top: dy, 
              width: dw, 
              height: dh,
              child: RawImage(
                image: _predictImage!,
                fit: BoxFit.contain,
                filterQuality: FilterQuality.none,
                isAntiAlias: false,
              ),
            ),
            Positioned(
              left: dx, 
              top: dy, 
              width: dw, 
              height: dh,
              child: CustomPaint(
                size: Size(dw, dh),
                painter: _AnnotationsPainter(
                  annotations: _annotations,
                  xScale: dw / imgW,
                  yScale: dh / imgH,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Future<void> _resetToCamera() async {
    setState(() {
      _isProcessing = false;
      _isCapturing = false;
      _processingProgress = 0.0;
      _processingStatus = null;
      _overlayImage = null;
      _predictImage = null;
      _annotations = [];
      _imageWidth = 0; 
      _imageHeight = 0;
    });

    await SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp
    ]);

    if (_isCameraReady) {
      await _controller.dispose();
    }

    _controller = CameraController(
      widget.cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await _controller.initialize();
    try { 
      await _controller.setFlashMode(FlashMode.off); 
    } catch (_) {}

    if (mounted) {
      setState(() {
        _isCameraReady = true;
        _currentState = AppPageState.camera;
      });
    }
  }
}

class _AnnotationsPainter extends CustomPainter {
  final List<Map<String, dynamic>> annotations;
  final double xScale;
  final double yScale;

  _AnnotationsPainter({
    required this.annotations,
    required this.xScale,
    required this.yScale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (annotations.isEmpty) return;

    for (final annotation in annotations) {
      final double cx = (annotation['x'] as num).toDouble() * xScale;
      final double cy = (annotation['y'] as num).toDouble() * yScale;
      final String text = ((annotation['class'] ?? '') as String).toUpperCase();

      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final pad = 8.0;
      final rect = RRect.fromRectAndRadius(
        Rect.fromLTWH(
          cx - tp.width / 2 - pad,
          cy - tp.height / 2 - pad,
          tp.width + pad * 2,
          tp.height + pad * 2,
        ),
        const Radius.circular(8),
      );

      final bgPaint = Paint()
        ..color = const Color(0xFF000000).withValues(alpha: 0.8)
        ..style = PaintingStyle.fill;
      
      final borderPaint = Paint()
        ..color = Colors.white.withValues(alpha: 0.3)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      canvas.drawRRect(rect, bgPaint);
      canvas.drawRRect(rect, borderPaint);

      tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
    }
  }

  @override
  bool shouldRepaint(covariant _AnnotationsPainter oldDelegate) {
    return oldDelegate.annotations != annotations ||
        oldDelegate.xScale != xScale ||
        oldDelegate.yScale != yScale;
  }
}
