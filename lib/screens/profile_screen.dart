import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:provider/provider.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
FlutterLocalNotificationsPlugin();

Future<void> initNotifications() async {
  const AndroidInitializationSettings initializationSettingsAndroid =
  AndroidInitializationSettings('@mipmap/ic_launcher');
  const InitializationSettings initializationSettings =
  InitializationSettings(android: initializationSettingsAndroid);
  await flutterLocalNotificationsPlugin.initialize(initializationSettings);
}

Future<void> showReminderNotification() async {
  const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
    'reminder_channel',
    'Nhắc nhở',
    channelDescription: 'Thông báo nhắc nhở kiểm tra mắt',
    importance: Importance.max,
    priority: Priority.high,
  );
  const NotificationDetails platformDetails =
  NotificationDetails(android: androidDetails);
  await flutterLocalNotificationsPlugin.show(
    0,
    'Nhắc nhở kiểm tra mắt',
    'Đã đến lúc bạn nên kiểm tra lại thị lực.',
    platformDetails,
  );
}

class ThemeProvider with ChangeNotifier {
  ThemeMode _themeMode = ThemeMode.light;
  ThemeMode get themeMode => _themeMode;

  void toggleTheme() {
    _themeMode =
    _themeMode == ThemeMode.light ? ThemeMode.dark : ThemeMode.light;
    notifyListeners();
  }
}

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  Future<void> backupTestResult(String result) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      await FirebaseFirestore.instance.collection('test_results').add({
        'uid': user.uid,
        'result': result,
        'timestamp': DateTime.now(),
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);

    return Scaffold(
      appBar: AppBar(title: const Text("Cài đặt")),
      body: ListView(
        children: [
          ListTile(
            leading: const Icon(Icons.notifications),
            title: const Text("Thông báo"),
            onTap: () async {
              await showReminderNotification();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Đã bật thông báo nhắc nhở")),
              );
            },
          ),
          ListTile(
            leading: const Icon(Icons.color_lens),
            title: const Text("Chế độ tối/sáng"),
            onTap: () {
              themeProvider.toggleTheme();
            },
          ),
          ListTile(
            leading: const Icon(Icons.privacy_tip),
            title: const Text("Chính sách riêng tư"),
            onTap: () {
              Navigator.push(context,
                  MaterialPageRoute(builder: (_) => const PrivacyScreen()));
            },
          ),
          ListTile(
            leading: const Icon(Icons.backup),
            title: const Text("Sao lưu dữ liệu"),
            onTap: () async {
              await backupTestResult("Kết quả test giả định: 20/40");
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("Đã sao lưu dữ liệu lên server")),
              );
            },
          ),
        ],
      ),
    );
  }
}

class PrivacyScreen extends StatelessWidget {
  const PrivacyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Chính sách riêng tư")),
      body: const Padding(
        padding: EdgeInsets.all(16),
        child: Text(
          "Ứng dụng cam kết bảo mật dữ liệu và không chia sẻ thông tin cá nhân của bạn cho bên thứ ba.",
          style: TextStyle(fontSize: 16),
        ),
      ),
    );
  }
}
