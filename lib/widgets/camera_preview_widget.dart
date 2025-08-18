import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class CameraPreviewWidget extends StatefulWidget {
  final CameraController controller;

  const CameraPreviewWidget({super.key, required this.controller});

  @override
  State<CameraPreviewWidget> createState() => _CameraPreviewWidgetState();
}

class _CameraPreviewWidgetState extends State<CameraPreviewWidget> {
  final bool _isRecording = false;
  final List<String> _capturedImages = [];
  Timer? _captureTimer;
  bool _isCapturing = false;
  
  @override
  void initState() {
    super.initState();
    _startEyeTracking();
  }

  @override
  void dispose() {
    _captureTimer?.cancel();
    super.dispose();
  }

  void _startEyeTracking() {
    _captureTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      _captureFrame();
    });
  }

  Future<void> _captureFrame() async {
    if (_isCapturing) return; // Skip if already capturing
    
    try {
      if (widget.controller.value.isInitialized) {
        _isCapturing = true;
        
        final Directory appDir = await getApplicationDocumentsDirectory();
        final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
        final String filePath = path.join(appDir.path, 'eye_frames', '$timestamp.jpg');
        
        await Directory(path.dirname(filePath)).create(recursive: true);
        
        final XFile imageFile = await widget.controller.takePicture();
        await imageFile.saveTo(filePath);
        
        setState(() {
          _capturedImages.add(filePath);
        });
        
        if (_capturedImages.length > 100) {
          final oldImage = _capturedImages.removeAt(0);
          try {
            await File(oldImage).delete();
          } catch (e) {
            print('Error deleting old image: $e');
          }
        }
      }
    } catch (e) {
      print('Error capturing frame: $e');
    } finally {
      _isCapturing = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 120,
      width: double.infinity,
      child: widget.controller.value.isInitialized
          ? Stack(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: CameraPreview(widget.controller),
                ),
                Positioned(
                  top: 8,
                  right: 8,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.fiber_manual_record,
                          color: Colors.red,
                          size: 12,
                        ),
                        const SizedBox(width: 4),
                        const Text(
                          'Recording',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Positioned(
                  bottom: 8,
                  left: 8,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      'Frames: ${_capturedImages.length}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ),
                Positioned.fill(
                  child: CustomPaint(
                    painter: EyeTrackingOverlayPainter(),
                  ),
                ),
              ],
            )
          : const Center(
              child: CircularProgressIndicator(),
            ),
    );
  }
}

class EyeTrackingOverlayPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final centerX = size.width / 2;
    final centerY = size.height / 2;
    final eyeWidth = 40.0;
    final eyeHeight = 25.0;

    final leftEyeRect = Rect.fromCenter(
      center: Offset(centerX - 30, centerY - 10),
      width: eyeWidth,
      height: eyeHeight,
    );
    
    final rightEyeRect = Rect.fromCenter(
      center: Offset(centerX + 30, centerY - 10),
      width: eyeWidth,
      height: eyeHeight,
    );

    canvas.drawOval(leftEyeRect, paint);
    canvas.drawOval(rightEyeRect, paint);

    final pupilPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.fill;

    canvas.drawCircle(Offset(centerX - 30, centerY - 10), 3, pupilPaint);
    canvas.drawCircle(Offset(centerX + 30, centerY - 10), 3, pupilPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}