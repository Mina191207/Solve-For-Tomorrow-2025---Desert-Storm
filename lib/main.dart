import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/auth_screen.dart';
import 'package:firebase_core/firebase_core.dart'; // Don't forget this import!
import 'firebase_options.dart'; // And this import for your generated options!
import 'package:firebase_auth/firebase_auth.dart';
import 'screens/main_navigation.dart';

final FirebaseAuth _auth = FirebaseAuth.instance;
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
  }
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  User? user = FirebaseAuth.instance.currentUser;

  runApp(VisionTestApp(isLoggedIn: user != null));
}

class VisionTestApp extends StatelessWidget {
  final bool isLoggedIn;
  const VisionTestApp({super.key, required this.isLoggedIn});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ứng dụng Kiểm tra Thị lực',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: 'Roboto',
      ),
      home: isLoggedIn ? const MainNavigation() : const AuthScreen(), // Your main screen where authentication happens
    );
  }
}

