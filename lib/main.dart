import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:image/image.dart' as img;

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    cameras = await availableCameras();
  } catch (_) {
    cameras = [];
  }

  runApp(const HandSignApp());
}

class HandSignApp extends StatelessWidget {
  const HandSignApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hand Sign Learner',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.indigo,
        scaffoldBackgroundColor: const Color(0xFFF6F7FB),
      ),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  // ============================================================
  // MODEL SETTINGS
  // ============================================================

  static const int inputSize = 224;
  static const int numClasses = 5;
  static const double confidenceThreshold = 0.70;

  // ============================================================
  // CAMERA / CLASSIFIER
  // ============================================================

  CameraController? _controller;

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;

  List<String> _labels = [];

  CameraLensDirection _currentLensDirection =
      CameraLensDirection.front;

  // ============================================================
  // HAND LANDMARKER
  // ============================================================

  HandLandmarkerPlugin? _handLandmarker;

  StreamSubscription<List<Hand>>? _landmarkSubscription;

  List<Hand> _detectedHands = [];

  bool _handDetected = false;

  // Prevent processing every single camera frame.
  bool _processingLandmarkFrame = false;

  DateTime _lastLandmarkTime = DateTime.fromMillisecondsSinceEpoch(0);

  static const Duration _landmarkInterval =
      Duration(milliseconds: 80);

  // ============================================================
  // APP STATE
  // ============================================================

  bool _modelReady = false;
  bool _cameraReady = false;
  bool _isRecognizing = false;
  bool _isSwitchingCamera = false;

  String _statusMessage = 'Preparing camera...';

  String _prediction = '';
  double _confidence = 0.0;

  // ============================================================
  // INITIALIZATION
  // ============================================================

  @override
  void initState() {
    super.initState();

    _initialize();
  }

  Future<void> _initialize() async {
    await _initializeHandLandmarker();

    await Future.wait([
      _loadModel(),
      _initCamera(),
    ]);
  }

  // ============================================================
  // INITIALIZE MEDIA PIPE HAND LANDMARKER
  // ============================================================

  Future<void> _initializeHandLandmarker() async {
    try {
      final landmarker = HandLandmarkerPlugin.create(
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        delegate: HandLandmarkerDelegate.cpu,
      );

      _handLandmarker = landmarker;

      _landmarkSubscription =
          landmarker.landmarkStream.listen(
        (hands) {
          if (!mounted) return;

          setState(() {
            _detectedHands = hands;
            _handDetected = hands.isNotEmpty;
          });
        },
        onError: (error) {
          if (!mounted) return;

          setState(() {
            _handDetected = false;
            _detectedHands = [];
          });
        },
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _statusMessage =
            'Unable to initialize hand tracking.\n\n$e';
      });
    }
  }

  // ============================================================
  // LOAD TENSORFLOW LITE CLASSIFIER
  // ============================================================

  Future<void> _loadModel() async {
    try {
      final interpreter = await Interpreter.fromAsset(
        'assets/model.tflite',
      );

      _interpreter = interpreter;

      _isolateInterpreter = await IsolateInterpreter.create(
        address: interpreter.address,
      );

      final labelsData = await rootBundle.loadString(
        'assets/labels.txt',
      );

      _labels = labelsData
          .split('\n')
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      if (!mounted) return;

      setState(() {
        _modelReady = true;

        if (_cameraReady) {
          _statusMessage = 'Ready';
        }
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _statusMessage =
            'Unable to load the recognition model.\n\n$e';
      });
    }
  }

  // ============================================================
  // INITIALIZE CAMERA
  // ============================================================

  Future<void> _initCamera() async {
    if (cameras.isEmpty) {
      if (!mounted) return;

      setState(() {
        _statusMessage =
            'No camera was found on this device.';
      });

      return;
    }

    final camera = cameras.firstWhere(
      (camera) =>
          camera.lensDirection ==
          CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    await _startCamera(camera);
  }

  // ============================================================
  // START CAMERA
  // ============================================================

  Future<void> _startCamera(
    CameraDescription camera,
  ) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _cameraReady = true;
        _currentLensDirection =
            camera.lensDirection;
        _detectedHands = [];
        _handDetected = false;

        if (_modelReady) {
          _statusMessage = 'Ready';
        } else {
          _statusMessage =
              'Loading recognition model...';
        }
      });

      // Start live camera frames for MediaPipe.
      await _startLandmarkStream(controller);
    } catch (e) {
      await controller.dispose();

      if (!mounted) return;

      setState(() {
        _cameraReady = false;
        _statusMessage =
            'Could not start the camera.\n\n$e';
      });
    }
  }

  // ============================================================
  // START LANDMARK STREAM
  // ============================================================

  Future<void> _startLandmarkStream(
    CameraController controller,
  ) async {
    if (!controller.value.isInitialized) return;

    if (controller.value.isStreamingImages) {
      return;
    }

    try {
      await controller.startImageStream(
        (CameraImage image) {
          _processLandmarkFrame(
            image,
            controller.description.sensorOrientation,
          );
        },
      );
    } catch (e) {
      debugPrint(
        'Could not start landmark camera stream: $e',
      );
    }
  }

  // ============================================================
  // PROCESS LIVE FRAME
  // ============================================================

  void _processLandmarkFrame(
    CameraImage image,
    int sensorOrientation,
  ) {
    final landmarker = _handLandmarker;

    if (landmarker == null) return;

    if (_isRecognizing || _isSwitchingCamera) {
      return;
    }

    if (_processingLandmarkFrame) {
      return;
    }

    final now = DateTime.now();

    if (now.difference(_lastLandmarkTime) <
        _landmarkInterval) {
      return;
    }

    _lastLandmarkTime = now;
    _processingLandmarkFrame = true;

    try {
      landmarker.processFrame(
        image,
        sensorOrientation,
      );
    } catch (e) {
      debugPrint(
        'Landmark processing error: $e',
      );
    } finally {
      _processingLandmarkFrame = false;
    }
  }

  // ============================================================
  // SWITCH CAMERA
  // ============================================================

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera || _isRecognizing) {
      return;
    }

    if (cameras.length < 2) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'This device does not have another camera.',
          ),
        ),
      );

      return;
    }

    setState(() {
      _isSwitchingCamera = true;
      _cameraReady = false;
      _statusMessage = 'Switching camera...';
      _prediction = '';
      _confidence = 0.0;
      _detectedHands = [];
      _handDetected = false;
    });

    try {
      final oldController = _controller;

      if (oldController != null) {
        try {
          if (oldController.value.isStreamingImages) {
            await oldController.stopImageStream();
          }
        } catch (_) {}

        await oldController.dispose();
      }

      _controller = null;

      final newCamera = cameras.firstWhere(
        (camera) =>
            camera.lensDirection !=
            _currentLensDirection,
        orElse: () => cameras.first,
      );

      await _startCamera(newCamera);

      if (!mounted) return;

      setState(() {
        _isSwitchingCamera = false;

        if (_modelReady) {
          _statusMessage = 'Ready';
        }
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isSwitchingCamera = false;
        _cameraReady = false;
        _statusMessage =
            'Could not switch camera.\n\n$e';
      });
    }
  }

  // ============================================================
  // SHUTTER BUTTON
  // ============================================================

  Future<void> _recognizeHandSign() async {
    if (_isRecognizing) return;

    if (!_modelReady ||
        _isolateInterpreter == null ||
        _labels.isEmpty) {
      _showMessage(
        'The recognition model is still loading.',
      );

      return;
    }

    final controller = _controller;

    if (controller == null ||
        !controller.value.isInitialized) {
      _showMessage(
        'The camera is not ready yet.',
      );

      return;
    }

    setState(() {
      _isRecognizing = true;
      _prediction = '';
      _confidence = 0.0;
      _statusMessage = 'Recognizing...';
      _detectedHands = [];
      _handDetected = false;
    });

    try {
      // ----------------------------------------------------------
      // Stop live stream before takePicture().
      // ----------------------------------------------------------

      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }

      // ----------------------------------------------------------
      // Capture a single image.
      // ----------------------------------------------------------

      final XFile photo =
          await controller.takePicture();

      // ----------------------------------------------------------
      // Read captured image.
      // ----------------------------------------------------------

      final Uint8List bytes =
          await photo.readAsBytes();

      // ----------------------------------------------------------
      // Decode image.
      // ----------------------------------------------------------

      final img.Image? decoded =
          img.decodeImage(bytes);

      if (decoded == null) {
        throw Exception(
          'Could not read the captured image.',
        );
      }

      // ----------------------------------------------------------
      // Fix orientation.
      // ----------------------------------------------------------

      final oriented =
          img.bakeOrientation(decoded);

      // ----------------------------------------------------------
      // Prepare image.
      // ----------------------------------------------------------

      final input =
          _preprocessPhoto(oriented);

      // ----------------------------------------------------------
      // Model output.
      // ----------------------------------------------------------

      final output = [
        List<double>.filled(
          numClasses,
          0.0,
        ),
      ];

      // ----------------------------------------------------------
      // Run TensorFlow Lite.
      // ----------------------------------------------------------

      await _isolateInterpreter!.run(
        input,
        output,
      );

      final scores = output[0];

      // ----------------------------------------------------------
      // Find highest score.
      // ----------------------------------------------------------

      int bestIndex = 0;
      double bestScore = scores[0];

      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];
          bestIndex = i;
        }
      }

      if (!mounted) return;

      // ----------------------------------------------------------
      // Result.
      // ----------------------------------------------------------

      if (bestScore >= confidenceThreshold &&
          bestIndex < _labels.length) {
        setState(() {
          _prediction =
              _friendlyLabel(_labels[bestIndex]);

          _confidence = bestScore;

          _statusMessage =
              'Recognition complete!';
        });
      } else {
        setState(() {
          _prediction = 'No clear sign';
          _confidence = bestScore;

          _statusMessage =
              'Try placing your hand inside the guide.';
        });
      }
    } catch (e) {
      debugPrint(
        'Recognition error: $e',
      );

      if (!mounted) return;

      setState(() {
        _prediction = '';
        _confidence = 0.0;

        _statusMessage =
            'Something went wrong.\n\nPlease try again.';
      });
    } finally {
      // ----------------------------------------------------------
      // Restart live landmarks.
      // ----------------------------------------------------------

      if (mounted) {
        setState(() {
          _isRecognizing = false;
        });

        await _startLandmarkStream(controller);
      }
    }
  }

  // ============================================================
  // PREPROCESS CAPTURED PHOTO
  // ============================================================

  List<List<List<List<double>>>> _preprocessPhoto(
    img.Image source,
  ) {
    final resized = img.copyResize(
      source,
      width: inputSize,
      height: inputSize,
    );

    return [
      List.generate(
        inputSize,
        (y) {
          return List.generate(
            inputSize,
            (x) {
              final pixel =
                  resized.getPixel(x, y);

              return [
                pixel.r.toDouble(),
                pixel.g.toDouble(),
                pixel.b.toDouble(),
              ];
            },
          );
        },
      ),
    ];
  }

  // ============================================================
  // FRIENDLY LABELS
  // ============================================================

  String _friendlyLabel(String label) {
    switch (label.toLowerCase().trim()) {
      case 'fist':
        return '✊ Fist';

      case 'ok_sign':
        return '👌 OK Sign';

      case 'open_palm':
        return '✋ Open Palm';

      case 'peace':
        return '✌️ Peace';

      case 'thumbs_up':
        return '👍 Thumbs Up';

      default:
        return label;
    }
  }

  // ============================================================
  // TRY AGAIN
  // ============================================================

  void _tryAgain() {
    if (!mounted) return;

    setState(() {
      _prediction = '';
      _confidence = 0.0;
      _statusMessage = 'Ready';
    });

    _startLandmarkStream(
      _controller!,
    );
  }

  // ============================================================
  // MESSAGE
  // ============================================================

  void _showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  @override
  void dispose() {
    _landmarkSubscription?.cancel();

    _handLandmarker?.dispose();

    _controller?.dispose();

    _isolateInterpreter?.close();

    _interpreter?.close();

    super.dispose();
  }

  // ============================================================
  // BUILD
  // ============================================================

  @override
  Widget build(BuildContext context) {
    final controller = _controller;

    final bool showResult =
        _prediction.isNotEmpty &&
        !_isRecognizing;

    // ----------------------------------------------------------
    // CAMERA NOT READY
    // ----------------------------------------------------------

    if (controller == null ||
        !controller.value.isInitialized) {
      return Scaffold(
        backgroundColor:
            const Color(0xFFF6F7FB),
        body: SafeArea(
          child: Center(
            child: Padding(
              padding:
                  const EdgeInsets.all(28),
              child: Column(
                mainAxisSize:
                    MainAxisSize.min,
                children: [
                  Container(
                    width: 82,
                    height: 82,
                    decoration:
                        BoxDecoration(
                      color:
                          Colors.indigo.shade50,
                      shape:
                          BoxShape.circle,
                    ),
                    child:
                        const Icon(
                      Icons
                          .front_hand_rounded,
                      size: 42,
                      color:
                          Colors.indigo,
                    ),
                  ),

                  const SizedBox(
                    height: 24,
                  ),

                  const Text(
                    'Hand Sign Learner',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight:
                          FontWeight.bold,
                    ),
                  ),

                  const SizedBox(
                    height: 12,
                  ),

                  Text(
                    _statusMessage,
                    textAlign:
                        TextAlign.center,
                    style: TextStyle(
                      fontSize: 15,
                      color:
                          Colors.grey.shade700,
                    ),
                  ),

                  const SizedBox(
                    height: 24,
                  ),

                  const CircularProgressIndicator(),
                ],
              ),
            ),
          ),
        ),
      );
    }

    // ----------------------------------------------------------
    // MAIN CAMERA SCREEN
    // ----------------------------------------------------------

    return Scaffold(
      backgroundColor: Colors.black,

      body: Stack(
        fit: StackFit.expand,
        children: [
          // ======================================================
          // CAMERA
          // ======================================================

          CameraPreview(controller),

          // ======================================================
          // LANDMARK OVERLAY
          // ======================================================

          IgnorePointer(
            child: CustomPaint(
              painter: HandLandmarkPainter(
                hands: _detectedHands,
                mirror:
                    _currentLensDirection ==
                        CameraLensDirection.front,
              ),
            ),
          ),

          // ======================================================
          // TOP GRADIENT
          // ======================================================

          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 165,
              decoration:
                  const BoxDecoration(
                gradient:
                    LinearGradient(
                  begin:
                      Alignment.topCenter,
                  end:
                      Alignment.bottomCenter,
                  colors: [
                    Colors.black54,
                    Colors.transparent,
                  ],
                ),
              ),
            ),
          ),

          // ======================================================
          // APP TITLE
          // ======================================================

          SafeArea(
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(
                horizontal: 18,
                vertical: 12,
              ),
              child: Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration:
                        BoxDecoration(
                      color:
                          Colors.white.withOpacity(
                        0.90,
                      ),
                      shape:
                          BoxShape.circle,
                    ),
                    child:
                        const Icon(
                      Icons
                          .front_hand_rounded,
                      color:
                          Colors.indigo,
                    ),
                  ),

                  const SizedBox(
                    width: 12,
                  ),

                  const Expanded(
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Hand Sign Learner',
                          style: TextStyle(
                            color:
                                Colors.white,
                            fontSize: 20,
                            fontWeight:
                                FontWeight.bold,
                          ),
                        ),

                        Text(
                          'Show your hand sign',
                          style: TextStyle(
                            color:
                                Colors.white70,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Camera switch
                  Material(
                    color: Colors.black
                        .withOpacity(0.45),
                    shape:
                        const CircleBorder(),
                    child: IconButton(
                      tooltip:
                          'Switch Camera',

                      onPressed:
                          (_isSwitchingCamera ||
                                  _isRecognizing)
                              ? null
                              : _switchCamera,

                      icon:
                          const Icon(
                        Icons
                            .flip_camera_ios_rounded,
                        color:
                            Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // ======================================================
          // HAND GUIDE + DETECTION STATUS
          // ======================================================

          Center(
            child: Column(
              mainAxisSize:
                  MainAxisSize.min,
              children: [
                AnimatedContainer(
                  duration:
                      const Duration(
                    milliseconds: 200,
                  ),
                  width: 245,
                  height: 300,
                  decoration:
                      BoxDecoration(
                    border: Border.all(
                      color: _handDetected
                          ? Colors.greenAccent
                          : Colors.white
                              .withOpacity(
                              0.85,
                            ),
                      width: 3,
                    ),
                    borderRadius:
                        BorderRadius.circular(
                      28,
                    ),
                    boxShadow:
                        _handDetected
                            ? [
                                BoxShadow(
                                  color: Colors
                                      .greenAccent
                                      .withOpacity(
                                    0.45,
                                  ),
                                  blurRadius: 18,
                                  spreadRadius: 2,
                                ),
                              ]
                            : null,
                  ),
                  child: Center(
                    child: _isRecognizing
                        ? const Column(
                            mainAxisSize:
                                MainAxisSize
                                    .min,
                            children: [
                              CircularProgressIndicator(
                                color:
                                    Colors.white,
                                strokeWidth:
                                    4,
                              ),

                              SizedBox(
                                height: 18,
                              ),

                              Text(
                                'Recognizing...',
                                style:
                                    TextStyle(
                                  color:
                                      Colors.white,
                                  fontSize:
                                      18,
                                  fontWeight:
                                      FontWeight
                                          .bold,
                                  shadows: [
                                    Shadow(
                                      blurRadius:
                                          6,
                                      color:
                                          Colors.black,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          )
                        : Text(
                            _handDetected
                                ? 'Hand detected ✓'
                                : 'Place your hand here',
                            style:
                                TextStyle(
                              color:
                                  _handDetected
                                      ? Colors
                                          .greenAccent
                                      : Colors
                                          .white,
                              fontSize: 17,
                              fontWeight:
                                  FontWeight.bold,
                              shadows: const [
                                Shadow(
                                  blurRadius:
                                      6,
                                  color:
                                      Colors.black,
                                ),
                              ],
                            ),
                          ),
                  ),
                ),

                const SizedBox(
                  height: 14,
                ),

                if (!_isRecognizing)
                  Container(
                    padding:
                        const EdgeInsets
                            .symmetric(
                      horizontal: 16,
                      vertical: 9,
                    ),
                    decoration:
                        BoxDecoration(
                      color: Colors.black
                          .withOpacity(
                        0.50,
                      ),
                      borderRadius:
                          BorderRadius.circular(
                        20,
                      ),
                    ),
                    child: Text(
                      _handDetected
                          ? 'Hand landmarks are tracking'
                          : 'Keep your hand inside the box',
                      style:
                          const TextStyle(
                        color:
                            Colors.white,
                        fontSize: 13,
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // ======================================================
          // HAND DETECTION STATUS PILL
          // ======================================================

          Positioned(
            top: 105,
            left: 0,
            right: 0,
            child: Center(
              child: AnimatedContainer(
                duration:
                    const Duration(
                  milliseconds: 200,
                ),
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 7,
                ),
                decoration:
                    BoxDecoration(
                  color: _handDetected
                      ? Colors.green
                      : Colors.black
                          .withOpacity(
                          0.55,
                        ),
                  borderRadius:
                      BorderRadius.circular(
                    20,
                  ),
                ),
                child: Row(
                  mainAxisSize:
                      MainAxisSize.min,
                  children: [
                    Icon(
                      _handDetected
                          ? Icons
                              .check_circle
                          : Icons
                              .pan_tool_outlined,
                      color:
                          Colors.white,
                      size: 17,
                    ),

                    const SizedBox(
                      width: 7,
                    ),

                    Text(
                      _handDetected
                          ? 'Hand detected'
                          : 'Searching for hand...',
                      style:
                          const TextStyle(
                        color:
                            Colors.white,
                        fontSize: 12,
                        fontWeight:
                            FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // ======================================================
          // BOTTOM AREA
          // ======================================================

          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              padding:
                  const EdgeInsets.fromLTRB(
                24,
                22,
                24,
                30,
              ),
              decoration:
                  BoxDecoration(
                gradient:
                    LinearGradient(
                  begin:
                      Alignment.topCenter,
                  end:
                      Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black
                        .withOpacity(
                      0.88,
                    ),
                  ],
                ),
              ),
              child: Column(
                children: [
                  // =================================================
                  // RESULT
                  // =================================================

                  if (showResult)
                    Container(
                      width:
                          double.infinity,
                      margin:
                          const EdgeInsets.only(
                        bottom: 20,
                      ),
                      padding:
                          const EdgeInsets.all(
                        20,
                      ),
                      decoration:
                          BoxDecoration(
                        color:
                            Colors.white,
                        borderRadius:
                            BorderRadius
                                .circular(
                          24,
                        ),
                        boxShadow: const [
                          BoxShadow(
                            blurRadius: 18,
                            offset:
                                Offset(
                              0,
                              8,
                            ),
                            color:
                                Colors.black26,
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          Container(
                            width: 58,
                            height: 58,
                            decoration:
                                BoxDecoration(
                              color:
                                  _prediction ==
                                          'No clear sign'
                                      ? Colors
                                          .orange
                                          .shade50
                                      : Colors
                                          .green
                                          .shade50,
                              shape:
                                  BoxShape
                                      .circle,
                            ),
                            child: Icon(
                              _prediction ==
                                      'No clear sign'
                                  ? Icons
                                      .help_outline_rounded
                                  : Icons
                                      .check_rounded,
                              color:
                                  _prediction ==
                                          'No clear sign'
                                      ? Colors
                                          .orange
                                      : Colors
                                          .green,
                              size: 34,
                            ),
                          ),

                          const SizedBox(
                            height: 10,
                          ),

                          Text(
                            _prediction ==
                                    'No clear sign'
                                ? 'Try Again'
                                : 'Great Job! 🎉',
                            style:
                                const TextStyle(
                              fontSize: 22,
                              fontWeight:
                                  FontWeight
                                      .bold,
                            ),
                          ),

                          const SizedBox(
                            height: 6,
                          ),

                          Text(
                            _prediction,
                            textAlign:
                                TextAlign
                                    .center,
                            style:
                                const TextStyle(
                              fontSize: 30,
                              fontWeight:
                                  FontWeight
                                      .w800,
                            ),
                          ),

                          if (_confidence >
                              0)
                            const SizedBox(
                              height: 5,
                            ),

                          if (_confidence >
                              0)
                            Text(
                              'Confidence: '
                              '${(_confidence * 100).toStringAsFixed(1)}%',
                              style:
                                  TextStyle(
                                color: Colors
                                    .grey
                                    .shade600,
                                fontSize: 14,
                              ),
                            ),

                          const SizedBox(
                            height: 16,
                          ),

                          SizedBox(
                            width:
                                double.infinity,
                            child:
                                FilledButton
                                    .icon(
                              onPressed:
                                  _tryAgain,
                              icon:
                                  const Icon(
                                Icons
                                    .refresh_rounded,
                              ),
                              label:
                                  const Text(
                                'Try Again',
                              ),
                              style:
                                  FilledButton
                                      .styleFrom(
                                padding:
                                    const EdgeInsets
                                        .symmetric(
                                  vertical:
                                      14,
                                ),
                                shape:
                                    RoundedRectangleBorder(
                                  borderRadius:
                                      BorderRadius
                                          .circular(
                                    16,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                  // =================================================
                  // SHUTTER BUTTON
                  // =================================================

                  if (!showResult)
                    Row(
                      mainAxisAlignment:
                          MainAxisAlignment
                              .center,
                      children: [
                        GestureDetector(
                          onTap:
                              _isRecognizing
                                  ? null
                                  : _recognizeHandSign,

                          child:
                              AnimatedContainer(
                            duration:
                                const Duration(
                              milliseconds:
                                  150,
                            ),
                            width: 82,
                            height: 82,
                            decoration:
                                BoxDecoration(
                              shape:
                                  BoxShape
                                      .circle,
                              color:
                                  _isRecognizing
                                      ? Colors
                                          .grey
                                      : Colors
                                          .white,
                              border:
                                  Border.all(
                                color:
                                    Colors
                                        .white,
                                width: 5,
                              ),
                              boxShadow:
                                  const [
                                BoxShadow(
                                  color: Colors
                                      .black45,
                                  blurRadius:
                                      12,
                                  offset:
                                      Offset(
                                    0,
                                    5,
                                  ),
                                ),
                              ],
                            ),
                            child:
                                Container(
                              margin:
                                  const EdgeInsets
                                      .all(
                                6,
                              ),
                              decoration:
                                  BoxDecoration(
                                shape:
                                    BoxShape
                                        .circle,
                                color:
                                    _isRecognizing
                                        ? Colors
                                            .grey
                                            .shade400
                                        : Colors
                                            .indigo,
                              ),
                              child:
                                  Icon(
                                _isRecognizing
                                    ? Icons
                                        .hourglass_top_rounded
                                    : Icons
                                        .camera_alt_rounded,
                                color:
                                    Colors
                                        .white,
                                size: 31,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),

                  const SizedBox(
                    height: 8,
                  ),

                  Text(
                    _isRecognizing
                        ? 'Analyzing your hand sign...'
                        : _handDetected
                            ? 'Your hand is ready — tap the camera'
                            : 'Show your hand inside the guide',
                    textAlign:
                        TextAlign.center,
                    style:
                        const TextStyle(
                      color:
                          Colors.white,
                      fontSize: 13,
                      fontWeight:
                          FontWeight.w500,
                      shadows: [
                        Shadow(
                          blurRadius: 5,
                          color:
                              Colors.black,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ==================================================================
// HAND LANDMARK PAINTER
// ==================================================================

class HandLandmarkPainter
    extends CustomPainter {
  final List<Hand> hands;
  final bool mirror;

  HandLandmarkPainter({
    required this.hands,
    required this.mirror,
  });

  // MediaPipe hand landmark connections.
  static const List<List<int>> connections = [
    // Thumb
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    // Index finger
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    // Middle finger
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    // Ring finger
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    // Pinky
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    // Palm
    [5, 9],
    [9, 13],
    [13, 17],
  ];

  @override
  void paint(
    Canvas canvas,
    Size size,
  ) {
    if (hands.isEmpty) return;

    final bonePaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final pointPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    final pointBorderPaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    for (final hand in hands) {
      final points = <Offset>[];

      for (final landmark in hand.landmarks) {
        double x = landmark.x;
        final double y = landmark.y;

        if (mirror) {
          x = 1.0 - x;
        }

        points.add(
          Offset(
            x * size.width,
            y * size.height,
          ),
        );
      }

      // Draw bones first.
      for (final connection in connections) {
        final startIndex = connection[0];
        final endIndex = connection[1];

        if (startIndex >= points.length ||
            endIndex >= points.length) {
          continue;
        }

        canvas.drawLine(
          points[startIndex],
          points[endIndex],
          bonePaint,
        );
      }

      // Draw landmark points.
      for (final point in points) {
        canvas.drawCircle(
          point,
          7,
          pointPaint,
        );

        canvas.drawCircle(
          point,
          7,
          pointBorderPaint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(
    covariant HandLandmarkPainter oldDelegate,
  ) {
    return oldDelegate.hands != hands ||
        oldDelegate.mirror != mirror;
  }
}