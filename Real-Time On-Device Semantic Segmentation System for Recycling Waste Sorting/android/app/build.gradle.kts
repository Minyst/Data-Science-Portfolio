// android/app/build.gradle.kts  (CPU 전용 · FP32 · 시연 안정형 · R8 OFF)
plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.reco"

    // Flutter가 넘겨주는 SDK/NDK 버전 사용
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "com.example.reco"
        // CPU 전용이라 21로 충분 (tflite_flutter 0.11.0 최소 21)
        minSdk = maxOf(21, flutter.minSdkVersion)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // 메서드 수 초과에 대비해 멀티덱스 활성화
        multiDexEnabled = true

        // 단일 ABI 타겟팅으로 APK 용량 감소 (갤럭시탭 S7: arm64)
        ndk {
            abiFilters.add("arm64-v8a")
        }
    }

    buildTypes {
        release {
            // ⚠️ 실제 배포 시 release 키로 교체
            signingConfig = signingConfigs.getByName("debug")
            // R8/ProGuard 비활성화 (안정성 우선)
            isMinifyEnabled = false
            isShrinkResources = false
        }
        debug {
            // 기본 설정 사용
        }
    }

    // splits ABI는 사용하지 않음 (ndk.abiFilters와 충돌 방지)

    // (필요시) 중복 리소스 충돌 회피
    packagingOptions {
        resources {
            excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // ✅ CPU 전용: GPU delegate 의존성 추가하지 않음
    // implementation("org.tensorflow:tensorflow-lite-gpu:2.xx.x")  <-- 제거
    // implementation("org.tensorflow:tensorflow-lite-gpu-api:2.xx.x")  <-- 제거

    // 멀티덱스 활성화를 위한 의존성
    implementation("androidx.multidex:multidex:2.0.1")

    // ONNX Runtime Android (안정 버전)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")
}
