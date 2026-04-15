import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'main_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  FlutterError.onError = (FlutterErrorDetails details) {
    FlutterError.dumpErrorToConsole(details);
  };
  // 비동기 에러도 삼켜서 앱이 종료되지 않도록 방지
  WidgetsBinding.instance.platformDispatcher.onError = (error, stack) {
    // 필요시 로깅/리포팅 추가 가능
    return true; // 에러를 처리했다고 알림 → 크래시 방지
  };
  final cameras = await availableCameras();
  runZonedGuarded(() {
    runApp(MyApp(cameras: cameras));
  }, (error, stack) {
    // 전역 예외 포착 (릴리즈에서 조용히 셧다운되는 문제 방지)
    // 필요시 네이티브 로그/서버 전송 가능
  });
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Reco',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF4CAF50)),
        useMaterial3: true,
      ),
      home: MainPage(cameras: cameras),
    );
  }
}
