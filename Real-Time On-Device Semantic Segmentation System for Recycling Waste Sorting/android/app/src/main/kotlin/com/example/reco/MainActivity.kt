package com.example.reco

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.TensorInfo

class MainActivity : FlutterActivity() {
    private val channelName = "onnx"
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null
    private var inputShape: LongArray? = null
    private var outputShape: LongArray? = null
    // 재사용 버퍼로 첫/이후 추론 속도 편차 최소화
    private var reorderBuffer: FloatArray? = null
    private var outputBuffer: FloatArray? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call: MethodCall, result: MethodChannel.Result ->
                when (call.method) {
                    "init" -> {
                        val assetKey = call.argument<String>("asset")
                            ?: return@setMethodCallHandler result.error("ARG", "missing asset", null)
                        // 무거운 초기화를 백그라운드에서 실행하여 ANR 방지
                        Thread {
                            try {
                                // Ensure JNI FindClass can locate ai.onnxruntime.* on this thread
                                Thread.currentThread().contextClassLoader = this@MainActivity.classLoader
                                initSessionFromAsset(assetKey)
                                val io = mapOf(
                                    "inputShape" to (inputShape?.map { it.toInt() } ?: emptyList()),
                                    "outputShape" to (outputShape?.map { it.toInt() } ?: emptyList()),
                                    "inputName" to (inputName ?: ""),
                                    "outputName" to (outputName ?: "")
                                )
                                runOnUiThread { result.success(io) }
                            } catch (e: Throwable) {
                                Log.e("ONNX", "init error", e)
                                runOnUiThread { result.error("ERR", e.message, null) }
                            }
                        }.start()
                    }
                    "run" -> {
                        val arr = call.argument<ByteArray>("input")
                            ?: return@setMethodCallHandler result.error("ARG", "missing input", null)
                        val inW = call.argument<Int>("width") ?: 0
                        val inH = call.argument<Int>("height") ?: 0
                        val inC = call.argument<Int>("channels") ?: 3
                        // 추론도 백그라운드에서 실행
                        Thread {
                            try {
                                Thread.currentThread().contextClassLoader = this@MainActivity.classLoader
                                val res = runInference(ByteBuffer.wrap(arr), inW, inH, inC)
                                runOnUiThread { result.success(res) }
                            } catch (e: Throwable) {
                                Log.e("ONNX", "run error", e)
                                runOnUiThread { result.error("ERR", e.message, null) }
                            }
                        }.start()
                    }
                    "warmup" -> {
                        // 입력 생성 없이 모델 고정 입력 크기로 1회 실행하여 커널/메모리 워밍업
                        Thread {
                            try {
                                Thread.currentThread().contextClassLoader = this@MainActivity.classLoader
                                warmupInference()
                                runOnUiThread { result.success(true) }
                            } catch (e: Throwable) {
                                Log.e("ONNX", "warmup error", e)
                                runOnUiThread { result.error("ERR", e.message, null) }
                            }
                        }.start()
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun initSessionFromAsset(assetKey: String) {
        // Flutter APK 내 자산 경로는 flutter_assets/ 접두사가 필요
        val fullAssetPath = if (assetKey.startsWith("flutter_assets/")) assetKey else "flutter_assets/$assetKey"
        // 자산 파일명(확장자 포함)을 그대로 보존하여 저장 (ORT/ONNX 구분)
        val assetFileName = File(assetKey).name
        val outFile = File(filesDir, assetFileName)
        // 과거 빌드에서 남은 model.onnx가 있다면 혼동 방지를 위해 삭제
        try {
            val legacy = File(filesDir, "model.onnx")
            if (legacy.exists() && !legacy.absolutePath.equals(outFile.absolutePath)) {
                legacy.delete()
            }
        } catch (_: Throwable) {}
        assets.open(fullAssetPath).use { input ->
            FileOutputStream(outFile).use { fos ->
                input.copyTo(fos)
            }
        }
        Log.i("ONNX", "Copy model to: ${outFile.absolutePath} (${outFile.length()} bytes)")
        env?.close()
        env = OrtEnvironment.getEnvironment()
        session?.close()
        val opts = SessionOptions()
        // 스레드 수 고정으로 실행 시간 변동 최소화 (일관성 향상)
        try {
            opts.setIntraOpNumThreads(4)
            opts.setInterOpNumThreads(1)
        } catch (_: Throwable) {}
        // ORT 포맷 명시 로딩 (ONNX Runtime Mobile이 ONNX 포맷을 미지원할 때를 대비)
        if (assetFileName.endsWith(".ort", ignoreCase = true)) {
            opts.addConfigEntry("session.load_model_format", "ORT")
        }
        // 모바일 빌드에서 미구현된 fused 커널 회피
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        Log.i("ONNX", "Loading model format: ${if (assetFileName.endsWith(".ort", true)) "ORT" else "ONNX"}")
        // Ensure ONNX runtime classes are resolvable on worker threads
        Thread.currentThread().contextClassLoader = this@MainActivity.classLoader
        session = env!!.createSession(outFile.absolutePath, opts)

        // 릴리즈에서 NodeInfo 생성 시 NoSuchMethodError 크래시가 발생하여
        // 메타데이터 조회를 우회한다. (모델은 고정 입력 1x3x512x512)
        inputName = "input"
        inputShape = longArrayOf(1, 3, 512, 512)
        // 입력 버퍼 재할당
        val inH = 512
        val inW = 512
        val inNum = inH * inW * 3
        reorderBuffer = FloatArray(inNum)

        // 출력 버퍼는 첫 실행 시 동적으로 크기 파악 후 할당 (runInference에서 처리)
        outputName = null
        outputShape = null
    }

    private fun runInference(inputBuffer: ByteBuffer, width: Int, height: Int, channels: Int): Map<String, Any> {
        requireNotNull(session) { "ONNX session not initialized" }
        requireNotNull(inputShape) { "Input shape unknown" }
        val shape = inputShape!!

        inputBuffer.order(ByteOrder.nativeOrder())
        val fb: FloatBuffer = inputBuffer.asFloatBuffer()
        val num = width * height * channels
        val src = FloatArray(num)
        fb.get(src)

        // 입력 레이아웃 감지: [1,3,H,W] or [1,H,W,3]
        val isNCHW = shape.size == 4 && shape[1].toInt() == channels
        val inH = if (isNCHW) shape[2].toInt() else shape[1].toInt()
        val inW = if (isNCHW) shape[3].toInt() else shape[2].toInt()

        // 리오더링
        val ordered = reorderBuffer ?: FloatArray(num).also { reorderBuffer = it }
        if (isNCHW) {
            var idx = 0
            for (y in 0 until inH) {
                for (x in 0 until inW) {
                    val base = (y * inW + x) * channels
                    // src는 NHWC(RGB) 순서라고 가정
                    ordered[idx + 0] = src[base + 0]
                    ordered[idx + inH * inW] = src[base + 1]
                    ordered[idx + 2 * inH * inW] = src[base + 2]
                    idx += 1
                }
            }
        } else {
            // 이미 NHWC면 그대로
            System.arraycopy(src, 0, ordered, 0, num)
        }

        val tensorShape = if (isNCHW) longArrayOf(1, channels.toLong(), inH.toLong(), inW.toLong()) else longArrayOf(1, inH.toLong(), inW.toLong(), channels.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(ordered), tensorShape)
        val res = session!!.run(mapOf(inputName!! to inputTensor))
        
        // 결과 추출 - res.get(0)를 OnnxTensor로 캐스팅
        val outTensor = res.get(0) as? OnnxTensor 
            ?: throw Exception("Output is not OnnxTensor: ${res.get(0)?.javaClass}")
        
        val outShapeArr = outTensor.info.shape
        val outBuf = outTensor.floatBuffer
        val outNum = outBuf.remaining()
        val outData = outputBuffer ?: FloatArray(outNum).also { outputBuffer = it }
        outBuf.get(outData)
        
        res.close()
        inputTensor.close()

        // 출력 [1,C,H,W] 또는 [1,H,W,C]를 [H,W,C]로 설명 반환
        val outIsNCHW = outShapeArr.size == 4 && outShapeArr[1] >= 1 && (outShapeArr[1] != outShapeArr[2] && outShapeArr[1] != outShapeArr[3])
        val outN = outShapeArr[0].toInt()
        val outC = if (outIsNCHW) outShapeArr[1].toInt() else outShapeArr[3].toInt()
        val outH = if (outIsNCHW) outShapeArr[2].toInt() else outShapeArr[1].toInt()
        val outW = if (outIsNCHW) outShapeArr[3].toInt() else outShapeArr[2].toInt()

        val hwc = FloatArray(outH * outW * outC)
        if (outIsNCHW) {
            // NCHW -> HWC
            var dst = 0
            for (y in 0 until outH) {
                for (x in 0 until outW) {
                    val idxHW = y * outW + x
                    for (c in 0 until outC) {
                        val srcIdx = c * (outH * outW) + idxHW
                        hwc[dst++] = outData[srcIdx]
                    }
                }
            }
        } else {
            // NHWC -> HWC (배치 제거)
            System.arraycopy(outData, 0, hwc, 0, hwc.size)
        }

        val shapeHWC = intArrayOf(outH, outW, outC)
        return mapOf(
            "data" to hwc.toList(),
            "shape" to shapeHWC.map { it }
        )
    }

    // 고정 입력 형태로 1회 워밍업 실행 (결과는 폐기)
    private fun warmupInference() {
        requireNotNull(session) { "ONNX session not initialized" }
        requireNotNull(inputShape) { "Input shape unknown" }
        val shape = inputShape!!
        val isNCHW = shape.size == 4 && shape[1].toInt() == 3
        val inH = if (isNCHW) shape[2].toInt() else shape[1].toInt()
        val inW = if (isNCHW) shape[3].toInt() else shape[2].toInt()
        val channels = 3
        val num = inW * inH * channels
        val zeros = FloatArray(num) { 0f }
        val tensorShape = if (isNCHW) longArrayOf(1, channels.toLong(), inH.toLong(), inW.toLong()) else longArrayOf(1, inH.toLong(), inW.toLong(), channels.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(zeros), tensorShape)
        val res = session!!.run(mapOf(inputName!! to inputTensor))
        // 자원 정리
        res.close()
        inputTensor.close()
    }
}
