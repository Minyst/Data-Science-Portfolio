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
  double _maskOpacity = 0.6;

  List<List<int>>? _lastLabels;
  List<List<int>>? _baseLabels; // 경계 정리 직후의 기준 라벨 (직선화 재적용용)
  List<List<List<double>>>? _processedOutput; // Softmax 적용 이후 모델 출력 (HWC 확률)

  // UI 파라미터
  double _edgeSharpness = 2.5; // 업샘플 확률 샤프닝 지수 (경계 더 선명하게)
  int _lineLength = 5; // 직선화 커널 길이 (퍼짐 최소화)
  // can 전환 임계값을 대상별로 분리
  double _dominanceToPlastic = 0.52; // can→plastic 전환 임계값 (하향)
  double _dominanceToPaper = 0.45;   // can→paper 전환 임계값

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
      height: _modelInputSize
    );
    
    final input = Float32List(_modelInputSize * _modelInputSize * 3);
    int index = 0;
    
    for (int y = 0; y < _modelInputSize; y++) {
      for (int x = 0; x < _modelInputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[index++] = pixel.r / 255.0;
        input[index++] = pixel.g / 255.0;
        input[index++] = pixel.b / 255.0;
      }
    }
    
    debugPrint('🔧 전처리 완료: 0~1 정규화 (RGB)');
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
      H, 
      (_) => List.generate(
        W, 
        (_) => List.filled(C, 0.0)
      )
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
    
    // 출력값 범위 확인
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
    
    // 중앙 픽셀 값 확인
    final centerY = outH ~/ 2;
    final centerX = outW ~/ 2;
    debugPrint('🔍 중앙 픽셀($centerX, $centerY) 값:');
    for (int c = 0; c < numClasses; c++) {
      final val = output[centerY][centerX][c];
      debugPrint('  클래스 $c (${c < _classNames.length ? _classNames[c] : "unknown"}): ${val.toStringAsFixed(4)}');
    }
    
    // Softmax 적용 여부 확인
    bool needsSoftmax = false;
    final sampleSum = List.generate(
      numClasses, 
      (c) => output[centerY][centerX][c]
    ).reduce((a, b) => a + b);
    
    if (sampleSum < 0.9 || sampleSum > 1.1) {
      needsSoftmax = true;
      debugPrint('🔄 Softmax 필요: 합=${sampleSum.toStringAsFixed(4)}');
          } else {
      debugPrint('✅ Softmax 불필요: 합=${sampleSum.toStringAsFixed(4)}');
    }
    
    // Softmax 적용 (필요한 경우)
    List<List<List<double>>> processedOutput = output;
    if (needsSoftmax) {
      processedOutput = List.generate(outH, (y) => 
        List.generate(outW, (x) {
          final logits = output[y][x];
          final maxLogit = logits.reduce((a, b) => a > b ? a : b);
          final expSum = logits
            .map((l) => math.exp(l - maxLogit))
            .reduce((a, b) => a + b);
          return logits
            .map((l) => math.exp(l - maxLogit) / expSum)
            .toList();
        })
      );
    }
    // 추후 실시간 재계산을 위해 저장
    _processedOutput = processedOutput;
    
    // 클래스별 평균 확률
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
    
    // Bilinear 업샘플링 + Argmax (부드러운 경계) → 헬퍼 함수 사용
    final labels = _labelsFromProcessed(processedOutput, originalWidth, originalHeight);
    
    // 최종 통계
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

  // processedOutput(HWC 확률)로부터 현재 UI 파라미터(_edgeSharpness)를 사용해 라벨 생성
  List<List<int>> _labelsFromProcessed(
    List<List<List<double>>> processedOutput,
    int originalWidth,
    int originalHeight,
  ) {
    final outH = processedOutput.length;
    final outW = processedOutput[0].length;
    final numClasses = processedOutput[0][0].length;

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
        final rawProbs = List<double>.filled(numClasses, 0.0);

        for (int c = 0; c < numClasses; c++) {
          final p00 = processedOutput[y0][x0][c];
          final p10 = processedOutput[y0][x1][c];
          final p01 = processedOutput[y1][x0][c];
          final p11 = processedOutput[y1][x1][c];

          final top = p00 * wx0 + p10 * wx1;
          final bottom = p01 * wx0 + p11 * wx1;
          double p = top * wy0 + bottom * wy1;
          final double sharpness = _edgeSharpness.clamp(1.0, 3.0);
          p = math.pow(p, sharpness).toDouble();
          final double clamped = p.clamp(0.0, 1.0);
          rawProbs[c] = clamped; // 가중치 적용 전 원확률 보관
          probs[c] = clamped;
        }

        if (probs.length >= 6) {
          // 배경 강화, can 강력 억제, paper 우선 강화, plastic은 약간 억제
          probs[0] *= 0.90; // background 강화 → 배경 점 최소화
          probs[1] *= 0.30; // can 더 강력 억제
          probs[2] *= 0.10; // glass 더 강하게 억제
          probs[3] *= 1.70; // paper 강화
          probs[4] *= 0.90; // plastic 소폭 억제
          probs[5] *= 0.10; // vinyl 더 강하게 억제
        }

        // 가중치 적용 후 상위 2개 클래스 선정
        int top1 = 0;
        double top1Val = probs[0];
        int top2 = -1;
        double top2Val = double.negativeInfinity;
        for (int c = 1; c < numClasses; c++) {
          final v = probs[c];
          if (v > top1Val) {
            top2 = top1; top2Val = top1Val;
            top1 = c; top1Val = v;
          } else if (v > top2Val) {
            top2 = c; top2Val = v;
          }
        }

        // can/plastic/paper 간 애매하면 원확률(rawProbs) 기준으로 결정
        // plastic vs paper인 경우 plastic에 0.8 보정 상수 적용하여 paper 우선성 강화
        final Set<int> ambiguous = {1, 3, 4};
        int finalClass = top1;
        if (top2 != -1 && ambiguous.contains(top1) && ambiguous.contains(top2)) {
          const double ambiguousEps = 0.06; // 구분 애매 범위 확대
          if ((top1Val - top2Val).abs() < ambiguousEps) {
            double a = rawProbs[top1];
            double b = rawProbs[top2];
            const int canIdx = 1;
            const int paperIdx = 3;
            const int plasticIdx = 4;
            const double plasticVsPaperBias = 0.55; // plastic이 paper에 미치는 영향 완화
            const double canAgainstOthersBias = 0.50; // can을 paper/plastic 대비 강력 약화

            // plastic vs paper bias
            if ((top1 == plasticIdx && top2 == paperIdx) || (top1 == paperIdx && top2 == plasticIdx)) {
              if (top1 == plasticIdx) a *= plasticVsPaperBias; else b *= plasticVsPaperBias;
            }

            // can vs paper/plastic bias (can 약화)
            if ((top1 == canIdx && (top2 == paperIdx || top2 == plasticIdx)) ||
                (top2 == canIdx && (top1 == paperIdx || top1 == plasticIdx))) {
              if (top1 == canIdx) a *= canAgainstOthersBias; else b *= canAgainstOthersBias;
            }

            // can이 정말 강한 경우는 존중 (원확률이 매우 높고 타 클래스 낮음)
            if (top1 == canIdx || top2 == canIdx) {
              final double canProb = rawProbs[canIdx];
              final double paperProb = rawProbs[paperIdx];
              final double plasticProb = rawProbs[plasticIdx];
              if (canProb >= 0.85 && paperProb <= 0.20 && plasticProb <= 0.20) {
                finalClass = canIdx;
              } else {
                finalClass = a >= b ? top1 : top2;
              }
            } else {
              finalClass = a >= b ? top1 : top2;
            }
          }
        }
        labels[y][x] = finalClass;
      }
    }

    return labels;
  }

  // can이 주변이 plastic/paper인 영역에 절대 끼어들지 못하도록 강력 억제
  // - can 컴포넌트 단위로 주변(테두리) 다수 클래스를 계산해 paper/plastic이면 해당 클래스로 강제 전환
  // - 단, can 평균 신뢰도(원확률)가 매우 높은 경우는 유지
  void _suppressCanInPlasticPaper(List<List<int>> labels) {
    if (_processedOutput == null) return;

    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;

    final int bgIdx = 0;
    final int canIdx = _classNames.indexOf('can');
    final int paperIdx = _classNames.indexOf('paper');
    final int plasticIdx = _classNames.indexOf('plastic');
    if (canIdx < 0 || paperIdx < 0 || plasticIdx < 0) return;

    final visited = List.generate(h, (_) => List<bool>.filled(w, false));
    const dirs4 = [1,0,-1,0,1];

    final outH = _processedOutput!.length;
    final outW = _processedOutput![0].length;

    int toOutY(int y) => ((y * outH) / h).clamp(0, outH - 1).toInt();
    int toOutX(int x) => ((x * outW) / w).clamp(0, outW - 1).toInt();

    final minCanArea = math.max(24, (w * h * 0.0015).round());
    const double strongCanProbThreshold = 0.75;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (visited[y][x] || labels[y][x] != canIdx) continue;
        // BFS로 can 컴포넌트 수집
        final q = <List<int>>[]..add([x, y]);
        final comp = <List<int>>[];
        visited[y][x] = true;
        int borderPaper = 0, borderPlastic = 0, borderBg = 0;
        double canProbSum = 0.0;

        while (q.isNotEmpty) {
          final cur = q.removeLast();
          final cx = cur[0], cy = cur[1];
          comp.add([cx, cy]);

          final oy = toOutY(cy);
          final ox = toOutX(cx);
          canProbSum += _processedOutput![oy][ox][canIdx];

          for (int i = 0; i < 4; i++) {
            final nx = cx + dirs4[i];
            final ny = cy + dirs4[i+1];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            // 테두리 클래스 카운트
            if (labels[ny][nx] == paperIdx) borderPaper++;
            else if (labels[ny][nx] == plasticIdx) borderPlastic++;
            else if (labels[ny][nx] == bgIdx) borderBg++;

            if (!visited[ny][nx] && labels[ny][nx] == canIdx) {
              visited[ny][nx] = true;
              q.add([nx, ny]);
            }
          }
        }

        final area = comp.length;
        final avgCanProb = canProbSum / area;

        // 주변 다수 클래스 결정 (paper/plastic 우선)
        int surround = bgIdx;
        if (borderPaper >= borderPlastic && borderPaper > 0) surround = paperIdx;
        if (borderPlastic > borderPaper && borderPlastic > 0) surround = plasticIdx;

        // 강력 억제 규칙:
        // - 컴포넌트가 작거나(minCanArea 미만)
        // - 주변 다수가 paper/plastic이며(can이 진짜라면 보통 배경에 둘러싸임)
        // - can 평균 확률이 충분히 높지 않으면
        //   → 전체 컴포넌트를 주변 다수 클래스로 전환 (paper/plastic 우선, 없으면 배경)
        final bool smallCan = area < minCanArea;
        final bool surroundedByPP = (surround == paperIdx || surround == plasticIdx);
        final bool canNotStrong = avgCanProb < strongCanProbThreshold;

        if (smallCan || (surroundedByPP && canNotStrong)) {
          final int target = (surround == paperIdx || surround == plasticIdx) ? surround : bgIdx;
          for (final p in comp) {
            labels[p[1]][p[0]] = target;
          }
        }
      }
    }
  }

  // 배경에 튀는 작은 점(작은 섬) 제거: 소형 컴포넌트를 배경으로 강제 전환
  void _removeSmallIslands(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;
    const dirs4 = [1,0,-1,0,1];
    final visited = List.generate(h, (_) => List<bool>.filled(w, false));

    final int bgIdx = 0;
    final minAreaPx = math.max(16, (w * h * 0.0005).round());

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (visited[y][x]) continue;
        final cls = labels[y][x];
        visited[y][x] = true;
        if (cls == bgIdx) continue;

        final q = <List<int>>[]..add([x, y]);
        final comp = <List<int>>[];
        while (q.isNotEmpty) {
          final cur = q.removeLast();
          final cx = cur[0], cy = cur[1];
          comp.add([cx, cy]);
          for (int i = 0; i < 4; i++) {
            final nx = cx + dirs4[i];
            final ny = cy + dirs4[i+1];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            if (visited[ny][nx]) continue;
            if (labels[ny][nx] != cls) continue;
            visited[ny][nx] = true;
            q.add([nx, ny]);
          }
        }

        if (comp.length < minAreaPx) {
          for (final p in comp) {
            labels[p[1]][p[0]] = bgIdx;
          }
        }
      }
    }
  }

  img.Image _labelsToColorMask(List<List<int>> labels) {
    final h = labels.length;
    final w = labels[0].length;
    final mask = img.Image(width: w, height: h);
    
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final clsIdx = labels[y][x];
        if (clsIdx <= 0 || clsIdx >= _classNames.length) {
          mask.setPixel(x, y, img.ColorRgb8(0, 0, 0));
        } else {
          final name = _classNames[clsIdx];
          final color = _classColors[name] ?? Colors.white;
          mask.setPixel(
            x, y, 
            img.ColorRgb8(color.red, color.green, color.blue)
          );
        }
      }
    }
    
    return mask;
  }

  Future<ui.Image> _createPredictionImage(img.Image mask) async {
    final bytes = Uint8List.fromList(img.encodePng(mask));
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  // 모폴로지 연산 (침식/팽창)
  List<List<bool>> _morphologicalOperation(
    List<List<bool>> mask, 
    {required bool isErosion, required int kernelSize}
  ) {
    final h = mask.length;
    final w = mask[0].length;
    final result = List.generate(h, (_) => List<bool>.filled(w, false));
    final radius = kernelSize ~/ 2;
    
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        bool shouldSet = isErosion;
        
        for (int ky = -radius; ky <= radius; ky++) {
          for (int kx = -radius; kx <= radius; kx++) {
            final ny = y + ky;
            final nx = x + kx;
            
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) {
              if (isErosion) {
                shouldSet = false;
            break;
          }
            continue;
          }
          
            if (isErosion) {
              if (!mask[ny][nx]) {
                shouldSet = false;
                break;
              }
            } else {
              if (mask[ny][nx]) {
                shouldSet = true;
                break;
              }
            }
          }
          if (!shouldSet && isErosion) break;
        }
        
        result[y][x] = shouldSet;
      }
    }
    
    return result;
  }

  // 부드러운 경계 + 정리
  void _cleanupClassBoundaries(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;
    
    debugPrint('🔧 클래스별 경계 정리 시작');
    
    // 1단계: 각 클래스별 Opening + Closing
    for (int targetClass = 1; targetClass < _classNames.length; targetClass++) {
      var mask = List.generate(h, (y) => List<bool>.filled(w, false));
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
          mask[y][x] = (labels[y][x] == targetClass);
        }
      }
      
      // Opening (작은 노이즈 제거, 팽창으로 인한 퍼짐 최소화)
      mask = _morphologicalOperation(mask, isErosion: true, kernelSize: 2);
      mask = _morphologicalOperation(mask, isErosion: false, kernelSize: 2);
      
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          if (mask[y][x]) {
            labels[y][x] = targetClass;
          }
        }
      }
    }
    
    // 2단계: Majority 필터로 경계 부드럽게 (1패스, 강한 다수만 허용)
    final copy = List.generate(h, (y) => List<int>.from(labels[y]));
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        if (copy[y][x] == 0) continue; // 배경은 건너뜀
        
        // 주변 8픽셀 확인
        final counts = <int, int>{};
                  for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
            final cls = copy[y + dy][x + dx];
            counts[cls] = (counts[cls] ?? 0) + 1;
          }
        }
        
        // Majority 클래스 찾기 (현재 클래스를 기본으로 하되, 강한 다수(>=6)일 때만 변경)
        final int currentClass = copy[y][x];
        int majorityClass = currentClass;
        int maxCount = counts[currentClass] ?? 0;
        counts.forEach((cls, count) {
          if (cls != 0 && count > maxCount) {
            maxCount = count;
            majorityClass = cls;
          }
        });
        
        if (majorityClass != currentClass && maxCount >= 6) {
          labels[y][x] = majorityClass;
        } else {
          labels[y][x] = currentClass;
        }
      }
    }
    
    debugPrint('✅ 경계 정리 완료 (부드러움)');
  }

  // 가로/세로 방향 직선화 모폴로지 연산
  List<List<bool>> _dilateDirectional(
    List<List<bool>> mask, {
    required int length,
    required bool horizontal,
  }) {
    final h = mask.length;
    final w = mask[0].length;
    final out = List.generate(h, (_) => List<bool>.filled(w, false));
    final r = length ~/ 2;
    if (horizontal) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          bool v = false;
          for (int k = -r; k <= r; k++) {
            final nx = x + k;
            if (nx < 0 || nx >= w) continue;
            if (mask[y][nx]) { v = true; break; }
          }
          out[y][x] = v;
        }
      }
    } else {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          bool v = false;
          for (int k = -r; k <= r; k++) {
            final ny = y + k;
            if (ny < 0 || ny >= h) continue;
            if (mask[ny][x]) { v = true; break; }
          }
          out[y][x] = v;
        }
      }
    }
    return out;
  }

  List<List<bool>> _erodeDirectional(
    List<List<bool>> mask, {
    required int length,
    required bool horizontal,
  }) {
    final h = mask.length;
    final w = mask[0].length;
    final out = List.generate(h, (_) => List<bool>.filled(w, false));
    final r = length ~/ 2;
    if (horizontal) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          bool v = true;
          for (int k = -r; k <= r; k++) {
            final nx = x + k;
            if (nx < 0 || nx >= w) { v = false; break; }
            if (!mask[y][nx]) { v = false; break; }
          }
          out[y][x] = v;
        }
      }
    } else {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          bool v = true;
          for (int k = -r; k <= r; k++) {
            final ny = y + k;
            if (ny < 0 || ny >= h) { v = false; break; }
            if (!mask[ny][x]) { v = false; break; }
          }
          out[y][x] = v;
        }
      }
    }
    return out;
  }

  void _straightenBoundaries(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;
    // 각 클래스별로 가로/세로 클로징으로 선형화
    final int len = _lineLength.isOdd ? _lineLength : _lineLength + 1; // 선 길이 (홀수)
    for (int cls = 1; cls < _classNames.length; cls++) {
      var mask = List.generate(h, (y) => List<bool>.filled(w, false));
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          mask[y][x] = (labels[y][x] == cls);
        }
      }

      // Horizontal/Vertical opening으로 경계 퍼짐 억제
      mask = _erodeDirectional(mask, length: len, horizontal: true);
      mask = _dilateDirectional(mask, length: len, horizontal: true);
      mask = _erodeDirectional(mask, length: len, horizontal: false);
      mask = _dilateDirectional(mask, length: len, horizontal: false);

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          if (mask[y][x]) labels[y][x] = cls;
        }
      }
    }
  }

  // 종이(paper)와 플라스틱(plastic) 경계 전용 보정
  // - plastic은 세로 방향 침범을 억제하기 위해 미세 침식
  // - paper는 세로 방향으로 미세 팽창하여 경계를 보호
  // - 충돌 시 paper 우선 적용
  void _refinePaperPlasticBoundary(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;

    final int paperIdx = _classNames.indexOf('paper');
    final int plasticIdx = _classNames.indexOf('plastic');
    if (paperIdx < 0 || plasticIdx < 0) return;

    var paperMask = List.generate(h, (y) => List<bool>.filled(w, false));
    var plasticMask = List.generate(h, (y) => List<bool>.filled(w, false));
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final v = labels[y][x];
        if (v == paperIdx) paperMask[y][x] = true;
        if (v == plasticIdx) plasticMask[y][x] = true;
      }
    }

    // 세로 보정만 유지
    plasticMask = _erodeDirectional(plasticMask, length: 4, horizontal: false);
    paperMask = _dilateDirectional(paperMask, length: 4, horizontal: false);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final orig = labels[y][x];
        if (orig != paperIdx && orig != plasticIdx) continue;

        final bool p = paperMask[y][x];
        final bool pl = plasticMask[y][x];
        if (p && !pl) {
          labels[y][x] = paperIdx;
        } else if (pl && !p) {
          labels[y][x] = plasticIdx;
        } else if (p && pl) {
          labels[y][x] = paperIdx; // 충돌 시 paper 우선
        }
      }
    }
  }

  // 플라스틱 우세 시 캔 픽셀을 플라스틱으로 강제 변환
  // - 3x3 윈도우 내 (can+plastic) 픽셀 중 plastic 비율이 임계값(_dominanceToPlastic) 이상이면 중심을 plastic으로 설정
  void _enforcePlasticDominance(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;

    final int canIdx = _classNames.indexOf('can');
    final int plasticIdx = _classNames.indexOf('plastic');
    if (canIdx < 0 || plasticIdx < 0) return;

    final src = List.generate(h, (y) => List<int>.from(labels[y]));
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        if (src[y][x] != canIdx) continue;

        int cpTotal = 0; // can+plastic 개수
        int plasticCount = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            final v = src[y + dy][x + dx];
            if (v == canIdx || v == plasticIdx) {
              cpTotal++;
              if (v == plasticIdx) plasticCount++;
            }
          }
        }

        if (cpTotal > 0) {
          final ratio = plasticCount / cpTotal;
          if (ratio >= _dominanceToPlastic) {
            labels[y][x] = plasticIdx;
          }
        }
      }
    }
  }

  // 종이(paper) 우세 시 캔 픽셀을 종이로 강제 변환
  // - 3x3 윈도우 내 (can+paper) 픽셀 중 paper 비율이 임계값(_dominanceToPaper) 이상이면 중심을 paper로 설정
  void _enforcePaperDominance(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;

    final int canIdx = _classNames.indexOf('can');
    final int paperIdx = _classNames.indexOf('paper');
    if (canIdx < 0 || paperIdx < 0) return;

    final src = List.generate(h, (y) => List<int>.from(labels[y]));
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        if (src[y][x] != canIdx) continue;

        int cpTotal = 0; // can+paper 개수
        int paperCount = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            final v = src[y + dy][x + dx];
            if (v == canIdx || v == paperIdx) {
              cpTotal++;
              if (v == paperIdx) paperCount++;
            }
          }
        }

        if (cpTotal > 0) {
          final ratio = paperCount / cpTotal;
          if (ratio >= _dominanceToPaper) {
            labels[y][x] = paperIdx;
          }
        }
      }
    }
  }

  // plastic→paper 비대칭 우세 전환 (paper가 우세 시에만 전환), 임계값 0.7
  void _enforcePlasticPaperDominance(List<List<int>> labels) {
    final h = labels.length;
    if (h == 0) return;
    final w = labels[0].length;

    final int paperIdx = _classNames.indexOf('paper');
    final int plasticIdx = _classNames.indexOf('plastic');
    if (paperIdx < 0 || plasticIdx < 0) return;

    const double threshold = 0.60; // plastic→paper 전환 기준 추가 완화
    final src = List.generate(h, (y) => List<int>.from(labels[y]));

    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final v = src[y][x];
        if (v != paperIdx && v != plasticIdx) continue;

        int ppTotal = 0;
        int paperCount = 0;
        int plasticCount = 0;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            final n = src[y + dy][x + dx];
            if (n == paperIdx || n == plasticIdx) {
              ppTotal++;
              if (n == paperIdx) paperCount++; else plasticCount++;
            }
          }
        }

        if (ppTotal > 0) {
          final paperRatio = paperCount / ppTotal;
          if (paperRatio >= threshold) {
            labels[y][x] = paperIdx; // paper 우세 시에만 전환 허용
          }
        }
      }
    }
  }

  // (동시판단 함수 제거됨) plastic/paper는 서로 강제 전환하지 않음

  List<Map<String, dynamic>> _buildSimpleAnnotations(List<List<int>> labels) {
    final h = labels.length, w = labels[0].length;
    final visited = List.generate(h, (_) => List<bool>.filled(w, false));
    const dirs4 = [1,0,-1,0,1];
    final result = <Map<String, dynamic>>[];

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (visited[y][x]) continue;
        final cls = labels[y][x];
        visited[y][x] = true;
        if (cls == 0) continue;

        final q = <List<int>>[]..add([x, y]);
        final comp = <List<int>>[];
        double sumX = 0, sumY = 0;
        
        while (q.isNotEmpty) {
          final cur = q.removeLast();
          final cx = cur[0], cy = cur[1];
          comp.add([cx, cy]);
          sumX += cx; 
          sumY += cy;
          
          for (int i = 0; i < 4; i++) {
            final nx = cx + dirs4[i];
            final ny = cy + dirs4[i+1];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            if (visited[ny][nx]) continue;
            if (labels[ny][nx] != cls) continue;
            visited[ny][nx] = true; 
            q.add([nx, ny]);
          }
        }

        final minArea = (w * h * 0.005).round();
        if (comp.length < minArea) continue;

        final cx = (sumX / comp.length).round();
        final cy = (sumY / comp.length).round();
        result.add({
          'x': cx, 
          'y': cy, 
          'class': _classNames[cls],
          'confidence': 1.0,
        });
      }
    }
    
    return result;
  }

  Future<void> _captureAndProcess() async {
    if (!_controller.value.isInitialized || _isCapturing || !_isModelLoaded) {
      return;
    }
    
    setState(() { 
      _isCapturing = true; 
      _processingStatus = '📸 촬영 중'; 
      _processingProgress = 0.1; 
    });

    try {
      try { 
        await _controller.setFlashMode(FlashMode.off); 
      } catch (_) {}
      
      SystemSound.play(SystemSoundType.click);
      HapticFeedback.mediumImpact();

      debugPrint('📸 사진 촬영 시작');
      final XFile image = await _controller.takePicture();
      final imageBytes = await image.readAsBytes();
      debugPrint('📸 사진 촬영 완료 - 크기: ${imageBytes.length} bytes');

      setState(() { 
        _processingStatus = '🔧 이미지 전처리'; 
        _processingProgress = 0.25; 
      });

      debugPrint(' 이미지 디코딩 시작');
      final originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) throw Exception('이미지 디코딩 실패');
      
      _imageWidth = originalImage.width;
      _imageHeight = originalImage.height;
      debugPrint(' 이미지 디코딩 완료 - 크기: ${_imageWidth}x${_imageHeight}');

      setState(() { 
        _processingStatus = '🧠 AI 분석 중'; 
        _processingProgress = 0.5; 
      });

      debugPrint('🔧 이미지 전처리 시작');
      final input = _preprocessImage(originalImage);
      debugPrint('🔧 이미지 전처리 완료 - 입력 크기: ${input.length}');
      
      debugPrint('🧠 모델 추론 시작');
      final output = await _runInference(input);
      debugPrint('🧠 모델 추론 완료');
      
      setState(() { 
        _processingStatus = '🎨 결과 생성 중'; 
        _processingProgress = 0.8; 
      });

      debugPrint('🎨 결과 생성 시작');
      final originalBytes = Uint8List.fromList(img.encodeJpg(originalImage));
      final oCodec = await ui.instantiateImageCodec(originalBytes);
      _originalUiImage = (await oCodec.getNextFrame()).image;

      final labels = _createSegmentationMask(output, _imageWidth, _imageHeight);
      
      // 모폴로지 연산으로 경계 정리
      _cleanupClassBoundaries(labels);
      // 직선화로 클래스 간 경계 정돈
      _straightenBoundaries(labels);
      _refinePaperPlasticBoundary(labels);
      _enforcePlasticPaperDominance(labels);
      _enforcePlasticDominance(labels);
      _enforcePaperDominance(labels);
      // can의 paper/plastic 침범 강력 억제 및 배경 점 제거
      _suppressCanInPlasticPaper(labels);
      _removeSmallIslands(labels);
      
      _baseLabels = List.generate(labels.length, (y) => List<int>.from(labels[y]));
      _lastLabels = labels;

      _annotations = _buildSimpleAnnotations(labels);

      final maskImg = _labelsToColorMask(labels);
      _predictImage = await _createPredictionImage(maskImg);
      _overlayImage = null;
      
      debugPrint('🎨 결과 생성 완료');

      setState(() {
        _processingStatus = '✅ 완료';
        _processingProgress = 1.0;
        _currentState = AppPageState.result;
      });

    } catch (e, stackTrace) {
      debugPrint('❌ 처리 오류: $e');
      debugPrint('스택 트레이스: $stackTrace');
      _showSnackBar('이미지 처리 중 오류가 발생했습니다: ${e.toString()}');
    } finally {
      setState(() {
        _isProcessing = false;
        _isCapturing = false;
        _processingProgress = 0.0;
        _processingStatus = null;
      });
    }
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
              filterQuality: FilterQuality.high,
              isAntiAlias: true,
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
                  filterQuality: FilterQuality.high,
                  isAntiAlias: true,
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
                filterQuality: FilterQuality.high,
                isAntiAlias: true,
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
      final String text = ((annotation['class'] ?? '') as String)
        .toUpperCase();

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

      tp.paint(
        canvas, 
        Offset(cx - tp.width / 2, cy - tp.height / 2)
      );
    }
  }

  @override
  bool shouldRepaint(covariant _AnnotationsPainter oldDelegate) {
    return oldDelegate.annotations != annotations ||
        oldDelegate.xScale != xScale ||
        oldDelegate.yScale != yScale;
  }
}

