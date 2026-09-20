import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Could not get cameras: $e');
    cameras = [];
  }

  runApp(const HandSignApp());
}

// ============================================================================
// APP
// ============================================================================

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

// ============================================================================
// CAMERA SCREEN
// ============================================================================

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  // ==========================================================================
  // LANDMARK MODEL SETTINGS
  // ==========================================================================

  static const int numLandmarks = 21;
  static const int valuesPerLandmark = 3;
  static const int totalFeatures =
      numLandmarks * valuesPerLandmark;

  // Only show a result when confidence reaches 70%.
  static const double confidenceThreshold = 0.70;

  // IMPORTANT:
  // The Python training pipeline used:
  //
  // 1. Original x/y/z landmarks
  // 2. Subtract wrist
  // 3. Find maximum absolute coordinate
  // 4. Divide by that scale
  // 5. Flatten to 63 values
  //
  // It did NOT mirror X during normalization.
  static const bool mirrorLandmarksForModel = false;

  // ==========================================================================
  // CAMERA
  // ==========================================================================

  CameraController? _controller;

  CameraLensDirection _currentLensDirection =
      CameraLensDirection.front;

  // ==========================================================================
  // TFLITE / LITERT
  // ==========================================================================

  Interpreter? _interpreter;

  List<String> _labels = [];

  bool _modelReady = false;

  // ==========================================================================
  // MEDIAPIPE HAND LANDMARKER
  // ==========================================================================

  HandLandmarkerPlugin? _handLandmarker;

  StreamSubscription<List<Hand>>? _landmarkSubscription;

  List<Hand> _detectedHands = [];

  bool _handDetected = false;

  bool _processingLandmarkFrame = false;

  DateTime _lastLandmarkTime =
      DateTime.fromMillisecondsSinceEpoch(0);

  static const Duration _landmarkInterval =
      Duration(milliseconds: 80);

  // ==========================================================================
  // APP STATE
  // ==========================================================================

  bool _cameraReady = false;

  bool _isRecognizing = false;

  bool _isSwitchingCamera = false;

  String _statusMessage = 'Preparing camera...';

  String _prediction = '';

  double _confidence = 0.0;

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================

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

  // ==========================================================================
  // MEDIAPIPE INITIALIZATION
  // ==========================================================================

  Future<void> _initializeHandLandmarker() async {
    try {
      debugPrint('');
      debugPrint('========================================');
      debugPrint('INITIALIZING MEDIAPIPE');
      debugPrint('========================================');

      final landmarker = HandLandmarkerPlugin.create(
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        delegate: HandLandmarkerDelegate.cpu,
      );

      _handLandmarker = landmarker;

      _landmarkSubscription =
          landmarker.landmarkStream.listen(
        (hands) {
          if (!mounted) {
            return;
          }

          setState(() {
            _detectedHands = hands;
            _handDetected = hands.isNotEmpty;
          });
        },
        onError: (error) {
          debugPrint(
            'Hand landmark stream error: $error',
          );

          if (!mounted) {
            return;
          }

          setState(() {
            _detectedHands = [];
            _handDetected = false;
          });
        },
      );

      debugPrint('MediaPipe initialized successfully.');
      debugPrint('========================================');
      debugPrint('');
    } catch (e, stackTrace) {
      debugPrint('');
      debugPrint('========================================');
      debugPrint('MEDIAPIPE INITIALIZATION ERROR');
      debugPrint('========================================');
      debugPrint('$e');
      debugPrint('$stackTrace');
      debugPrint('========================================');

      if (!mounted) {
        return;
      }

      setState(() {
        _statusMessage =
            'Unable to initialize hand tracking.\n\n$e';
      });
    }
  }

  // ==========================================================================
  // LOAD LANDMARK TFLITE MODEL
  // ==========================================================================

  Future<void> _loadModel() async {
    try {
      debugPrint('');
      debugPrint('========================================');
      debugPrint('LOADING LANDMARK TFLITE MODEL');
      debugPrint('========================================');

      final interpreter = await Interpreter.fromAsset(
        'assets/model.tflite',
      );

      interpreter.allocateTensors();

      _interpreter = interpreter;

      // ----------------------------------------------------------------------
      // INPUT TENSOR
      // ----------------------------------------------------------------------

      final inputTensor =
          interpreter.getInputTensor(0);

      debugPrint('MODEL INPUT');
      debugPrint(
        '  Name : ${inputTensor.name}',
      );
      debugPrint(
        '  Shape: ${inputTensor.shape}',
      );
      debugPrint(
        '  Type : ${inputTensor.type}',
      );
      debugPrint(
        '  Bytes: ${inputTensor.numBytes()}',
      );

      // ----------------------------------------------------------------------
      // OUTPUT TENSOR
      // ----------------------------------------------------------------------

      final outputTensor =
          interpreter.getOutputTensor(0);

      debugPrint('MODEL OUTPUT');
      debugPrint(
        '  Name : ${outputTensor.name}',
      );
      debugPrint(
        '  Shape: ${outputTensor.shape}',
      );
      debugPrint(
        '  Type : ${outputTensor.type}',
      );
      debugPrint(
        '  Bytes: ${outputTensor.numBytes()}',
      );

      debugPrint(
        'LiteRT version: ${Interpreter.version}',
      );

      // ----------------------------------------------------------------------
      // LOAD LABELS
      // ----------------------------------------------------------------------

      final labelsData =
          await rootBundle.loadString(
        'assets/labels.txt',
      );

      _labels = labelsData
          .split(RegExp(r'\r?\n'))
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      debugPrint(
        'LABELS: $_labels',
      );

      if (_labels.isEmpty) {
        throw Exception(
          'labels.txt is empty.',
        );
      }

      // ----------------------------------------------------------------------
      // VERIFY INPUT SHAPE
      // ----------------------------------------------------------------------

      if (inputTensor.shape.length != 2 ||
          inputTensor.shape[0] != 1 ||
          inputTensor.shape[1] != totalFeatures) {
        throw Exception(
          'Unexpected model input shape: '
          '${inputTensor.shape}\n'
          'Expected: [1, $totalFeatures]',
        );
      }

      // ----------------------------------------------------------------------
      // VERIFY OUTPUT SHAPE
      // ----------------------------------------------------------------------

      if (outputTensor.shape.length != 2 ||
          outputTensor.shape[0] != 1 ||
          outputTensor.shape[1] != _labels.length) {
        throw Exception(
          'Unexpected model output shape: '
          '${outputTensor.shape}\n'
          'Expected: [1, ${_labels.length}]',
        );
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _modelReady = true;

        if (_cameraReady) {
          _statusMessage = 'Ready';
        }
      });

      debugPrint('');
      debugPrint('========================================');
      debugPrint('MODEL READY');
      debugPrint('========================================');
      debugPrint(
        'Input : ${inputTensor.shape}',
      );
      debugPrint(
        'Output: ${outputTensor.shape}',
      );
      debugPrint(
        'Labels: $_labels',
      );
      debugPrint('========================================');
      debugPrint('');
    } catch (e, stackTrace) {
      debugPrint('');
      debugPrint('========================================');
      debugPrint('MODEL LOAD ERROR');
      debugPrint('========================================');
      debugPrint('$e');
      debugPrint('$stackTrace');
      debugPrint('========================================');

      if (!mounted) {
        return;
      }

      setState(() {
        _modelReady = false;

        _statusMessage =
            'Unable to load the recognition model.\n\n$e';
      });
    }
  }

  // ==========================================================================
  // CAMERA INITIALIZATION
  // ==========================================================================

  Future<void> _initCamera() async {
    if (cameras.isEmpty) {
      if (!mounted) {
        return;
      }

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

  // ==========================================================================
  // START CAMERA
  // ==========================================================================

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

      await _startLandmarkStream(controller);
    } catch (e) {
      await controller.dispose();

      debugPrint(
        'Camera initialization error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _cameraReady = false;

        _statusMessage =
            'Could not start the camera.\n\n$e';
      });
    }
  }

  // ==========================================================================
  // START CAMERA IMAGE STREAM
  // ==========================================================================

  Future<void> _startLandmarkStream(
    CameraController controller,
  ) async {
    if (!controller.value.isInitialized) {
      return;
    }

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

      debugPrint(
        'Camera image stream started.',
      );
    } catch (e) {
      debugPrint(
        'Could not start landmark camera stream: $e',
      );
    }
  }

  // ==========================================================================
  // PROCESS LIVE CAMERA FRAME
  // ==========================================================================

  void _processLandmarkFrame(
    CameraImage image,
    int sensorOrientation,
  ) {
    final landmarker = _handLandmarker;

    if (landmarker == null) {
      return;
    }

    if (_isRecognizing ||
        _isSwitchingCamera) {
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

  // ==========================================================================
  // NORMALIZE LANDMARKS
  //
  // EXACTLY MATCHES THE PYTHON TRAINING PIPELINE:
  //
  // 21 landmarks × 3 = 63 values
  //
  // 1. Read x/y/z
  // 2. Subtract wrist coordinates
  // 3. Find maximum absolute value
  // 4. Divide all values by scale
  // 5. Flatten to 63 values
  // ==========================================================================

  List<double> _normalizeLandmarks(
    Hand hand,
  ) {
    if (hand.landmarks.length !=
        numLandmarks) {
      throw Exception(
        'Expected $numLandmarks hand landmarks, '
        'but received ${hand.landmarks.length}.',
      );
    }

    final values =
        List<List<double>>.generate(
      numLandmarks,
      (index) {
        final landmark =
            hand.landmarks[index];

        double x = landmark.x;

        final double y = landmark.y;

        final double z = landmark.z;

        // IMPORTANT:
        // This remains false because the Python training
        // normalization did not mirror X.
        if (mirrorLandmarksForModel) {
          x = 1.0 - x;
        }

        return [x, y, z];
      },
    );

    // ------------------------------------------------------------------------
    // WRIST
    // ------------------------------------------------------------------------

    final wristX = values[0][0];
    final wristY = values[0][1];
    final wristZ = values[0][2];

    // ------------------------------------------------------------------------
    // MOVE WRIST TO ORIGIN
    // ------------------------------------------------------------------------

    for (int i = 0;
        i < numLandmarks;
        i++) {
      values[i][0] -= wristX;
      values[i][1] -= wristY;
      values[i][2] -= wristZ;
    }

    // ------------------------------------------------------------------------
    // FIND MAXIMUM ABSOLUTE VALUE
    // ------------------------------------------------------------------------

    double scale = 0.0;

    for (int i = 0;
        i < numLandmarks;
        i++) {
      scale = math.max(
        scale,
        values[i][0].abs(),
      );

      scale = math.max(
        scale,
        values[i][1].abs(),
      );

      scale = math.max(
        scale,
        values[i][2].abs(),
      );
    }

    // ------------------------------------------------------------------------
    // SCALE NORMALIZATION
    // ------------------------------------------------------------------------

    if (scale > 1e-6) {
      for (int i = 0;
          i < numLandmarks;
          i++) {
        values[i][0] /= scale;
        values[i][1] /= scale;
        values[i][2] /= scale;
      }
    }

    // ------------------------------------------------------------------------
    // FLATTEN 21 × 3 INTO 63
    // ------------------------------------------------------------------------

    final flattened = <double>[];

    for (int i = 0;
        i < numLandmarks;
        i++) {
      flattened.add(values[i][0]);
      flattened.add(values[i][1]);
      flattened.add(values[i][2]);
    }

    if (flattened.length !=
        totalFeatures) {
      throw Exception(
        'Invalid landmark feature count: '
        '${flattened.length}. '
        'Expected $totalFeatures.',
      );
    }

    return flattened;
  }

  // ==========================================================================
  // RECOGNIZE HAND SIGN
  // ==========================================================================

  Future<void> _recognizeHandSign() async {
    if (_isRecognizing) {
      return;
    }

    // ------------------------------------------------------------------------
    // CHECK MODEL
    // ------------------------------------------------------------------------

    final interpreter = _interpreter;

    if (!_modelReady ||
        interpreter == null ||
        _labels.isEmpty) {
      _showMessage(
        'The recognition model is still loading.',
      );

      return;
    }

    // ------------------------------------------------------------------------
    // COPY CURRENT HAND
    // ------------------------------------------------------------------------

    final handsForRecognition =
        List<Hand>.from(
      _detectedHands,
    );

    if (handsForRecognition.isEmpty) {
      _showMessage(
        'No hand detected. Place your hand inside the guide first.',
      );

      return;
    }

    final hand =
        handsForRecognition.first;

    if (hand.landmarks.length !=
        numLandmarks) {
      _showMessage(
        'Could not read all 21 hand landmarks. Try again.',
      );

      return;
    }

    // ------------------------------------------------------------------------
    // RECOGNITION STATE
    // ------------------------------------------------------------------------

    setState(() {
      _isRecognizing = true;

      _prediction = '';

      _confidence = 0.0;

      _statusMessage = 'Recognizing...';
    });

    try {
      // ======================================================================
      // STEP 1: NORMALIZE LANDMARKS
      // ======================================================================

      final normalizedLandmarks =
          _normalizeLandmarks(hand);

      debugPrint('');
      debugPrint(
        '========================================',
      );
      debugPrint(
        'STARTING LANDMARK RECOGNITION',
      );
      debugPrint(
        '========================================',
      );

      debugPrint(
        'Landmarks received: '
        '${hand.landmarks.length}',
      );

      debugPrint(
        'Feature count: '
        '${normalizedLandmarks.length}',
      );

      debugPrint(
        'Features: ${normalizedLandmarks.map(
          (e) => e.toStringAsFixed(4),
        ).toList()}',
      );

      // ======================================================================
      // STEP 2: CREATE [1,63] INPUT
      // ======================================================================

      final input = <List<double>>[
        normalizedLandmarks,
      ];

      debugPrint(
        'TFLite input shape: '
        '[1, ${normalizedLandmarks.length}]',
      );

      // ======================================================================
      // STEP 3: CREATE [1,5] OUTPUT
      // ======================================================================

      final output =
          <List<double>>[
        List<double>.filled(
          _labels.length,
          0.0,
        ),
      ];

      debugPrint(
        'TFLite output buffer: '
        '[1, ${_labels.length}]',
      );

      // ======================================================================
      // STEP 4: RUN MODEL
      // ======================================================================

      debugPrint(
        'Running TFLite landmark model...',
      );

      interpreter.run(
        input,
        output,
      );

      debugPrint(
        'TFLite model finished.',
      );

      // ======================================================================
      // STEP 5: READ OUTPUT
      // ======================================================================

      final scores = output[0];

      debugPrint(
        'RAW OUTPUT: $scores',
      );

      debugPrint('');
      debugPrint(
        'PREDICTION SCORES:',
      );

      for (int i = 0;
          i < scores.length &&
              i < _labels.length;
          i++) {
        debugPrint(
          '  ${_labels[i]} = '
          '${scores[i].toStringAsFixed(6)} '
          '(${(scores[i] * 100).toStringAsFixed(2)}%)',
        );
      }

      // ======================================================================
      // STEP 6: FIND HIGHEST SCORE
      // ======================================================================

      int bestIndex = 0;

      double bestScore = scores[0];

      for (int i = 1;
          i < scores.length;
          i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];

          bestIndex = i;
        }
      }

      debugPrint('');
      debugPrint(
        'BEST INDEX: $bestIndex',
      );

      debugPrint(
        'BEST LABEL: '
        '${_labels[bestIndex]}',
      );

      debugPrint(
        'BEST SCORE: $bestScore',
      );

      debugPrint(
        'BEST CONFIDENCE: '
        '${(bestScore * 100).toStringAsFixed(2)}%',
      );

      debugPrint(
        '========================================',
      );
      debugPrint('');

      if (!mounted) {
        return;
      }

      // ======================================================================
      // STEP 7: DISPLAY RESULT
      // ======================================================================

      if (bestScore >= confidenceThreshold) {
        setState(() {
          _prediction =
              _friendlyLabel(
            _labels[bestIndex],
          );

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
    } catch (e, stackTrace) {
      debugPrint('');
      debugPrint(
        '========================================',
      );
      debugPrint(
        'RECOGNITION ERROR',
      );
      debugPrint(
        '========================================',
      );
      debugPrint('$e');
      debugPrint('$stackTrace');
      debugPrint(
        '========================================',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _prediction = '';

        _confidence = 0.0;

        _statusMessage =
            'Recognition failed.\n\n$e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isRecognizing = false;
        });
      }
    }
  }

  // ==========================================================================
  // FRIENDLY LABEL
  // ==========================================================================

  String _friendlyLabel(
    String label,
  ) {
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

  // ==========================================================================
  // TRY AGAIN
  // ==========================================================================

  void _tryAgain() {
    if (!mounted) {
      return;
    }

    setState(() {
      _prediction = '';

      _confidence = 0.0;

      _statusMessage =
          _modelReady && _cameraReady
              ? 'Ready'
              : 'Preparing...';
    });
  }

  // ==========================================================================
  // SWITCH CAMERA
  // ==========================================================================

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera ||
        _isRecognizing) {
      return;
    }

    if (cameras.length < 2) {
      ScaffoldMessenger.of(context)
          .showSnackBar(
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

      _statusMessage =
          'Switching camera...';

      _prediction = '';

      _confidence = 0.0;

      _detectedHands = [];

      _handDetected = false;
    });

    try {
      // ----------------------------------------------------------------------
      // STOP OLD CAMERA
      // ----------------------------------------------------------------------

      final oldController =
          _controller;

      if (oldController != null) {
        try {
          if (oldController
              .value
              .isStreamingImages) {
            await oldController.stopImageStream();
          }
        } catch (_) {}

        await oldController.dispose();
      }

      _controller = null;

      // ----------------------------------------------------------------------
      // FIND OTHER CAMERA
      // ----------------------------------------------------------------------

      final newCamera =
          cameras.firstWhere(
        (camera) =>
            camera.lensDirection !=
            _currentLensDirection,
        orElse: () => cameras.first,
      );

      await _startCamera(newCamera);

      if (!mounted) {
        return;
      }

      setState(() {
        _isSwitchingCamera = false;

        if (_modelReady) {
          _statusMessage = 'Ready';
        }
      });
    } catch (e) {
      debugPrint(
        'Camera switch error: $e',
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _isSwitchingCamera = false;

        _cameraReady = false;

        _statusMessage =
            'Could not switch camera.\n\n$e';
      });
    }
  }

  // ==========================================================================
  // SHOW MESSAGE
  // ==========================================================================

  void _showMessage(
    String message,
  ) {
    if (!mounted) {
      return;
    }

    ScaffoldMessenger.of(context)
        .showSnackBar(
      SnackBar(
        content: Text(message),
        behavior:
            SnackBarBehavior.floating,
      ),
    );
  }

  // ==========================================================================
  // DISPOSE
  // ==========================================================================

  @override
  void dispose() {
    _landmarkSubscription?.cancel();

    _handLandmarker?.dispose();

    _controller?.dispose();

    _interpreter?.close();

    super.dispose();
  }

  // ==========================================================================
  // BUILD
  // ==========================================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    final controller = _controller;

    final bool showResult =
        _prediction.isNotEmpty &&
        !_isRecognizing;

    // =========================================================================
    // CAMERA NOT READY
    // =========================================================================

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

    // =========================================================================
    // MAIN CAMERA SCREEN
    // =========================================================================

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ===================================================================
          // CAMERA + HAND LANDMARKS
          // ===================================================================

          Center(
            child: AspectRatio(
              aspectRatio:
                  controller
                          .value
                          .previewSize!
                          .height /
                      controller
                          .value
                          .previewSize!
                          .width,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(
                    controller,
                  ),

                  IgnorePointer(
                    child: CustomPaint(
                      painter:
                          HandLandmarkPainter(
                        hands:
                            _detectedHands,
                        previewSize:
                            controller
                                .value
                                .previewSize!,
                        lensDirection:
                            controller
                                .description
                                .lensDirection,
                        sensorOrientation:
                            controller
                                .description
                                .sensorOrientation,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // ===================================================================
          // TOP GRADIENT
          // ===================================================================

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

          // ===================================================================
          // APP TITLE
          // ===================================================================

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
                          Colors.white
                              .withValues(
                        alpha: 0.90,
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
                          CrossAxisAlignment
                              .start,
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

                  Material(
                    color: Colors.black
                        .withValues(
                      alpha: 0.45,
                    ),
                    shape:
                        const CircleBorder(),
                    child:
                        IconButton(
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

          // ===================================================================
          // HAND GUIDE
          // ===================================================================

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
                              .withValues(
                              alpha: 0.85,
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
                                      .withValues(
                                    alpha: 0.45,
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
                                MainAxisSize.min,
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
                              shadows:
                                  const [
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
                          .withValues(
                        alpha: 0.50,
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

          // ===================================================================
          // HAND DETECTION STATUS
          // ===================================================================

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
                          .withValues(
                          alpha: 0.55,
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

          // ===================================================================
          // BOTTOM AREA
          // ===================================================================

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
                        .withValues(
                      alpha: 0.88,
                    ),
                  ],
                ),
              ),
              child: Column(
                children: [
                  // ===========================================================
                  // RESULT CARD
                  // ===========================================================

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

                          if (_confidence > 0)
                            const SizedBox(
                              height: 5,
                            ),

                          if (_confidence > 0)
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

                  // ===========================================================
                  // SHUTTER BUTTON
                  // ===========================================================

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

// ============================================================================
// HAND LANDMARK PAINTER
// ============================================================================

class HandLandmarkPainter
    extends CustomPainter {
  HandLandmarkPainter({
    required this.hands,
    required this.previewSize,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  final List<Hand> hands;

  final Size previewSize;

  final CameraLensDirection
      lensDirection;

  final int sensorOrientation;

  // MediaPipe hand connections.
  static const List<List<int>>
      connections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    [5, 9],
    [9, 13],
    [13, 17],
  ];

  @override
  void paint(
    Canvas canvas,
    Size size,
  ) {
    if (hands.isEmpty ||
        size.width <= 0 ||
        size.height <= 0) {
      return;
    }

    final scale =
        size.width /
            previewSize.height;

    final pointPaint = Paint()
      ..color = Colors.greenAccent
      ..style =
          PaintingStyle.fill;

    final pointBorderPaint =
        Paint()
          ..color = Colors.white
          ..strokeWidth =
              2 / scale
          ..style =
              PaintingStyle.stroke;

    final linePaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth =
          4 / scale
      ..strokeCap =
          StrokeCap.round
      ..style =
          PaintingStyle.stroke;

    canvas.save();

    final center = Offset(
      size.width / 2,
      size.height / 2,
    );

    canvas.translate(
      center.dx,
      center.dy,
    );

    canvas.rotate(
      sensorOrientation *
          math.pi /
          180,
    );

    if (lensDirection ==
        CameraLensDirection.front) {
      canvas.scale(-1, 1);

      canvas.rotate(math.pi);
    }

    canvas.scale(scale);

    final logicalWidth =
        previewSize.width;

    final logicalHeight =
        previewSize.height;

    // ========================================================================
    // DRAW EVERY DETECTED HAND
    // ========================================================================

    for (final hand in hands) {
      final points =
          <Offset>[];

      // ----------------------------------------------------------------------
      // CREATE SCREEN POINTS
      // ----------------------------------------------------------------------

      for (final landmark
          in hand.landmarks) {
        final dx =
            (landmark.x - 0.5) *
                logicalWidth;

        final dy =
            (landmark.y - 0.5) *
                logicalHeight;

        points.add(
          Offset(dx, dy),
        );
      }

      // ----------------------------------------------------------------------
      // DRAW CONNECTIONS
      // ----------------------------------------------------------------------

      for (final connection
          in connections) {
        final startIndex =
            connection[0];

        final endIndex =
            connection[1];

        if (startIndex >=
                points.length ||
            endIndex >=
                points.length) {
          continue;
        }

        canvas.drawLine(
          points[startIndex],
          points[endIndex],
          linePaint,
        );
      }

      // ----------------------------------------------------------------------
      // DRAW 21 LANDMARK POINTS
      // ----------------------------------------------------------------------

      for (final point
          in points) {
        canvas.drawCircle(
          point,
          7 / scale,
          pointPaint,
        );

        canvas.drawCircle(
          point,
          7 / scale,
          pointBorderPaint,
        );
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(
    covariant HandLandmarkPainter
        oldDelegate,
  ) {
    return true;
  }
}