import 'package:flutter/material.dart';
import '../models/test_result.dart';
import '../models/test_session.dart';
import '../services/ml_service.dart';
import '../services/test_data_service.dart';
import '../services/camera_service.dart';
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:archive/archive_io.dart';
import 'package:path/path.dart' as path;
import 'package:http_parser/http_parser.dart';

class ResultsScreen extends StatefulWidget {
  final String testType;
  final List<TestResult> testResults;
  final DateTime testStartTime;

  const ResultsScreen({
    super.key,
    required this.testType,
    required this.testResults,
    required this.testStartTime,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  bool _isAnalyzing = true;
  VisionAnalysisResult? _analysisResult;
  final MLService _mlService = MLService();
  final TestSessionManager _sessionManager = TestSessionManager();
  final SessionStorage _sessionStorage = SessionStorage();
  final TestDataService _testDataService = TestDataService();
  final CameraService _cameraService = CameraService();

  @override
  void initState(){
    super.initState();
    // final stats = TestDataService().getTestStatistics();
    // final jsonString = const JsonEncoder.withIndent('  ').convert(stats);
    // final file = File('test_statistics.json');
    // file.writeAsString(jsonString);
    //
    // print('✅ Statistics saved at: ${file.path}');
    // saveStatistics(stats);
    // final loaded = loadStatistics();
    // print("✅ Loaded: $loaded");
    _analyzeResults();
  }

  // Future<File> _getLocalFile() async {
  //   // Get the app's document directory (safe & persistent)
  //   final directory = await getApplicationDocumentsDirectory();
  //   return File('${directory.path}/test_statistics.json');
  // }
  // Future<void> saveStatistics(Map<String, dynamic> stats) async {
  //   final file = await _getLocalFile();
  //   final jsonString = jsonEncode(stats);
  //   await file.writeAsString(jsonString);
  //   print("📂 Saved stats at: ${file.path}");
  // }

  // Future<Map<String, dynamic>> loadStatistics() async {
  //   final file = await _getLocalFile();
  //
  //   if (await file.exists()) {
  //     final contents = await file.readAsString();
  //     return jsonDecode(contents);
  //   } else {
  //     return {}; // return empty if file doesn’t exist
  //   }
  // }
  Future<File> zipEyeCapturesFolder() async {
    final appDir = await getApplicationDocumentsDirectory();
    final saveDir = Directory('${appDir.path}/eye_captures');
    final zipPath = path.join(appDir.path, 'eye_captures.zip');

    // ✅ Delete old ZIP if it exists
    final oldZip = File(zipPath);
    if (await oldZip.exists()) {
      await oldZip.delete();
      print('🗑️ Old ZIP deleted at $zipPath');
    }

    final encoder = ZipFileEncoder();
    encoder.create(zipPath);

    // ✅ Add all files in eye_captures folder
    saveDir.listSync().whereType<File>().forEach((file) {
      encoder.addFile(file);
    });

    encoder.close();
    print('📦 New ZIP created at $zipPath');

    return File(zipPath);
  }


  Future<void> uploadFolderAndJson() async {
    final appDir = await getApplicationDocumentsDirectory();
    final jsonFile = File(path.join(appDir.path, 'sessions.json'));
    final zipFile = await zipEyeCapturesFolder();

    final uri = Uri.parse("https://d80cf1f8c8b8.ngrok-free.app/upload");
    var request = http.MultipartRequest('POST', uri);

    // Add JSON
    request.files.add(await http.MultipartFile.fromPath(
      'session',
      jsonFile.path,
      contentType: MediaType('application', 'json'),
    ));

    // Add ZIP
    request.files.add(await http.MultipartFile.fromPath(
      'images_zip',
      zipFile.path,
      contentType: MediaType('application', 'zip'),
    ));

    final response = await request.send();

    if (response.statusCode == 200) {
      print("✅ Upload successful");
    } else {
      print("❌ Upload failed: ${response.statusCode}");
    }
  }

  Future<void> _analyzeResults() async {
    try {
      final currentSession = _sessionManager.getCurrentSession();
      if (currentSession != null) {
        _testDataService.addCompletedSession(currentSession);
        await SessionStorage.saveSessions([currentSession]);
        await CameraService().saveAllCapturedImages();
        final loaded = await SessionStorage.loadSessions();
        print("Loaded ${loaded.length} sessions");
        uploadFolderAndJson();
        
        EyeAnalysisResult? eyeAnalysis;
        if (_cameraService.hasCaptures()) {
          eyeAnalysis = _cameraService.getAggregateAnalysis();
        }
        
        final eyeTrackingData = _cameraService.generateEyeTrackingData();
        
        _mlService.analyzeVisionTest(
          widget.testType,
          widget.testResults,
          eyeTrackingData,
        ).then((mlResult) {
          print('ML Analysis completed (background): ${mlResult.diagnosis}');
          // Store for future use without displaying
        }).catchError((error) {
          print('ML Analysis error (background): $error');
        });
        
        final testBasedResult = _createTestBasedAnalysis();
        
        setState(() {
          _analysisResult = testBasedResult;
          _isAnalyzing = false;
        });
      } else {
        throw Exception('No current session found');
      }
    } catch (e) {
      print('Error analyzing results: $e');
      setState(() {
        _isAnalyzing = false;
      });
    }
  }

  VisionAnalysisResult _createTestBasedAnalysis() {
    final correctAnswers = widget.testResults.where((r) => r.isCorrect).length;
    final totalQuestions = widget.testResults.length;
    final accuracy = totalQuestions > 0 ? correctAnswers / totalQuestions : 0.0;
    
    // Determine risk level based on test performance only
    String riskLevel;
    String diagnosis;
    List<String> recommendations;
    
    if (accuracy >= 0.8) {
      riskLevel = 'Low';
      diagnosis = 'Hiệu suất kiểm tra xuất sắc. Thị lực có vẻ hoạt động tốt.';
      recommendations = [
        'Tiếp tục kiểm tra mắt định kỳ',
        'Duy trì thói quen sống lành mạnh',
        'Bảo vệ mắt khỏi bức xạ UV',
        'Tuân theo quy tắc 20-20-20 khi sử dụng thiết bị điện tử'
      ];
    } else if (accuracy >= 0.6) {
      riskLevel = 'Medium';
      diagnosis = 'Hiệu suất kiểm tra tốt với một số khu vực cần cải thiện.';
      recommendations = [
        'Lên lịch kiểm tra mắt toàn diện',
        'Theo dõi thay đổi thị lực theo thời gian',
        'Cân nhắc chỉnh sửa thị lực nếu cần',
        'Kiểm tra sức khỏe mắt định kỳ'
      ];
    } else {
      riskLevel = 'High';
      diagnosis = 'Hiệu suất kiểm tra chỉ ra các vấn đề thị lực tiềm ẩn.';
      recommendations = [
        'Lên lịch kiểm tra mắt toàn diện sớm',
        'Nên đánh giá chuyên nghiệp',
        'Theo dõi thay đổi thị lực chặt chẽ',
        'Cân nhắc các lựa chọn chỉnh sửa thị lực'
      ];
    }
    
    return VisionAnalysisResult(
      visionScore: accuracy,
      riskLevel: riskLevel,
      diagnosis: diagnosis,
      recommendations: recommendations,
      confidence: 0.85, // Base confidence on test reliability
      eyeAnalysis: null, // Don't include AI analysis in results
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Kết quả Kiểm tra'),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: _isAnalyzing ? _buildAnalyzingWidget() : _buildResultsWidget(),
    );
  }

  Widget _buildAnalyzingWidget() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text(
            'Đang phân tích kết quả kiểm tra...',
            style: TextStyle(fontSize: 16),
          ),
          SizedBox(height: 8),
          Text(
            'Đang tính điểm thị lực của bạn',
            style: TextStyle(fontSize: 14, color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsWidget() {
    if (_analysisResult == null) {
      return const Center(
        child: Text(
          'Không thể phân tích kết quả',
          style: TextStyle(fontSize: 16, color: Colors.red),
        ),
      );
    }

    final result = _analysisResult!;
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(),
          const SizedBox(height: 24),
          _buildScoreCard(result),
          const SizedBox(height: 24),
          _buildDiagnosisCard(result),
          const SizedBox(height: 24),
          _buildRecommendationsCard(result),
          const SizedBox(height: 24),
          _buildTestDetailsCard(),
          const SizedBox(height: 24),
          _buildActionButtons(),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.blue.shade500, Colors.blue.shade700],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            widget.testType,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Kiểm tra hoàn thành vào ${_formatDate(widget.testStartTime)}',
            style: const TextStyle(
              fontSize: 14,
              color: Colors.white70,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScoreCard(VisionAnalysisResult result) {
    Color scoreColor = result.riskLevel == 'Low' 
        ? Colors.green 
        : result.riskLevel == 'Medium' 
            ? Colors.orange 
            : Colors.red;

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Điểm Thị lực',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: LinearProgressIndicator(
                    value: result.visionScore,
                    backgroundColor: Colors.grey.shade300,
                    valueColor: AlwaysStoppedAnimation<Color>(scoreColor),
                    minHeight: 8,
                  ),
                ),
                const SizedBox(width: 16),
                Text(
                  '${(result.visionScore * 100).toInt()}%',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: scoreColor,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: scoreColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: scoreColor.withOpacity(0.3)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    result.riskLevel == 'Low' 
                        ? Icons.check_circle 
                        : result.riskLevel == 'Medium' 
                            ? Icons.warning 
                            : Icons.error,
                    size: 16,
                    color: scoreColor,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Rủi ro ${result.riskLevel}',
                    style: TextStyle(
                      color: scoreColor,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDiagnosisCard(VisionAnalysisResult result) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Phân tích Kiểm tra',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              result.diagnosis,
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.assessment, size: 16, color: Colors.blue),
                const SizedBox(width: 4),
                Text(
                  'Dựa trên hiệu suất kiểm tra',
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.blue,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAIAnalysisCard(EyeAnalysisResult eyeAnalysis) {
    Color conditionColor = eyeAnalysis.condition == 'normal' 
        ? Colors.green 
        : Colors.orange;

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.smart_toy,
                  color: Colors.blue.shade600,
                  size: 20,
                ),
                const SizedBox(width: 8),
                const Text(
                  'AI Eye Analysis',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: conditionColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: conditionColor.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  Icon(
                    eyeAnalysis.condition == 'normal' 
                        ? Icons.check_circle 
                        : Icons.warning,
                    color: conditionColor,
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Detected: ${eyeAnalysis.condition.replaceAll('_', ' ').toUpperCase()}',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: conditionColor,
                      ),
                    ),
                  ),
                  Text(
                    '${(eyeAnalysis.confidence * 100).toInt()}%',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: conditionColor,
                    ),
                  ),
                ],
              ),
            ),
            if (eyeAnalysis.riskFactors.isNotEmpty) ...[
              const SizedBox(height: 16),
              const Text(
                'Risk Factors:',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              ...eyeAnalysis.riskFactors.map((factor) => Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.fiber_manual_record,
                      size: 8,
                      color: Colors.grey.shade600,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        factor,
                        style: const TextStyle(fontSize: 13),
                      ),
                    ),
                  ],
                ),
              )),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationsCard(VisionAnalysisResult result) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Khuyến nghị',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...result.recommendations.map((recommendation) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.check_circle_outline, size: 16, color: Colors.green),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      recommendation,
                      style: const TextStyle(fontSize: 14),
                    ),
                  ),
                ],
              ),
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildTestDetailsCard() {
    final correctAnswers = widget.testResults.where((result) => result.isCorrect).length;
    final totalQuestions = widget.testResults.length;
    final accuracy = totalQuestions > 0 ? (correctAnswers / totalQuestions * 100).toInt() : 0;

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Chi tiết Kiểm tra',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Độ chính xác:'),
                Text(
                  '$accuracy%',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Câu trả lời Đúng:'),
                Text(
                  '$correctAnswers/$totalQuestions',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Thời gian Kiểm tra:'),
                Text(
                  '${DateTime.now().difference(widget.testStartTime).inMinutes} phút',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: ElevatedButton(
            onPressed: () {
              // Clear the current session since test is complete
              _sessionManager.clearSession();
              // Cleanup captured images and analysis data
              _cameraService.cleanup();
              // Navigate back to home screen
              Navigator.of(context).popUntil((route) => route.isFirst);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            child: const Text('Về Trang chủ'),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton(
            onPressed: () {
              _showShareDialog();
            },
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.blue,
              side: const BorderSide(color: Colors.blue),
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            child: const Text('Chia sẻ Kết quả'),
          ),
        ),
      ],
    );
  }

  void _showShareDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Chia sẻ Kết quả'),
        content: const Text(
          'Chia sẻ kết quả kiểm tra với chuyên gia y tế hoặc lưu lại để tham khảo sau này.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Hủy'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Kết quả đã được lưu vào thiết bị')),
              );
            },
            child: const Text('Lưu'),
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year} at ${date.hour}:${date.minute.toString().padLeft(2, '0')}';
  }
}