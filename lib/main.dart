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

  // Show a definite result when confidence reaches 70%.
  // Below that the card says "No clear sign" and the status text names the
  // closest match, so you can tell "guessing the right sign, low confidence"
  // apart from "guessing a completely different sign".
  static const double confidenceThreshold = 0.60;

  // The hand_landmarker plugin returns x/y in the camera SENSOR frame
  // (landscape, not rotated). HandLandmarkPainter rotates them for display;
  // _landmarksToFeatures() applies the same rotation so the model sees an
  // UPRIGHT hand, like the Python training pipeline did.
  //
  // We do not know whether the training frames were selfie-mirrored, so when
  // this is true we score both the on-screen view and its mirror image and keep
  // the more confident one. All five signs mean the same thing mirrored, and
  // it also handles left vs right hands.
  static const bool tryBothOrientations = true;

  // When Capture is pressed, average the model output over the last few
  // frames instead of trusting a single noisy frame.
  static const int _historyFrames = 5;
  static const Duration _historyWindow =
      Duration(milliseconds: 700);

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

  // Recent single-hand results (newest last), used to average on capture.
  final List<_TimedHand> _handHistory = <_TimedHand>[];

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

          if (hands.isNotEmpty) {
            _handHistory.add(
              _TimedHand(hands.first, DateTime.now()),
            );

            while (_handHistory.length > 12) {
              _handHistory.removeAt(0);
            }
          } else {
            _handHistory.clear();
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
      ResolutionPreset.medium,
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
  // LANDMARKS -> MODEL FEATURES
  //
  // 1. Convert the plugin's sensor-frame x/y into on-screen (upright) pixel
  //    coordinates. This is exactly the transform HandLandmarkPainter uses to
  //    draw the skeleton, so if the green overlay sits on your hand, the
  //    features are in the same orientation.
  // 2. Optionally mirror X.
  // 3. Subtract the wrist, divide by the largest absolute value, flatten to 63
  //    (same as the Python training pipeline).
  // ==========================================================================

  List<double> _landmarksToFeatures(
    Hand hand, {
    required bool mirror,
  }) {
    if (hand.landmarks.length != numLandmarks) {
      throw Exception(
        'Expected $numLandmarks hand landmarks, '
        'but received ${hand.landmarks.length}.',
      );
    }

    final controller = _controller;

    if (controller == null ||
        !controller.value.isInitialized ||
        controller.value.previewSize == null) {
      throw Exception('Camera is not ready.');
    }

    final Size previewSize = controller.value.previewSize!;
    final double sensorWidth = previewSize.width;
    final double sensorHeight = previewSize.height;

    final bool isFront =
        controller.description.lensDirection ==
            CameraLensDirection.front;

    final double angle =
        controller.description.sensorOrientation *
            math.pi /
            180.0;

    final double cosA = math.cos(angle);
    final double sinA = math.sin(angle);

    final values = List<List<double>>.generate(
      numLandmarks,
      (index) {
        final landmark = hand.landmarks[index];

        // Pixels relative to the frame centre (sensor frame).
        final double px =
            (landmark.x - 0.5) * sensorWidth;
        double py =
            (landmark.y - 0.5) * sensorHeight;

        // Painter: front camera = scale(-1, 1) then rotate(pi) => (px, -py).
        if (isFront) {
          py = -py;
        }

        // Painter: rotate by sensorOrientation.
        double sx = px * cosA - py * sinA;
        final double sy = px * sinA + py * cosA;

        // z is in roughly the same units as x (fraction of frame width).
        final double z = landmark.z * sensorWidth;

        if (mirror) {
          sx = -sx;
        }

        return [sx, sy, z];
      },
    );

    final wristX = values[0][0];
    final wristY = values[0][1];
    final wristZ = values[0][2];

    for (int i = 0; i < numLandmarks; i++) {
      values[i][0] -= wristX;
      values[i][1] -= wristY;
      values[i][2] -= wristZ;
    }

    double scale = 0.0;

    for (int i = 0; i < numLandmarks; i++) {
      scale = math.max(scale, values[i][0].abs());
      scale = math.max(scale, values[i][1].abs());
      scale = math.max(scale, values[i][2].abs());
    }

    if (scale > 1e-6) {
      for (int i = 0; i < numLandmarks; i++) {
        values[i][0] /= scale;
        values[i][1] /= scale;
        values[i][2] /= scale;
      }
    }

    final flattened = <double>[];

    for (int i = 0; i < numLandmarks; i++) {
      flattened.add(values[i][0]);
      flattened.add(values[i][1]);
      flattened.add(values[i][2]);
    }

    if (flattened.length != totalFeatures) {
      throw Exception(
        'Invalid landmark feature count: '
        '${flattened.length}. '
        'Expected $totalFeatures.',
      );
    }

    return flattened;
  }

  // ==========================================================================
  // RUN THE TFLITE MODEL ON ONE FEATURE VECTOR
  // ==========================================================================

  List<double> _runModel(
    Interpreter interpreter,
    List<double> features,
  ) {
    final input = <List<double>>[features];

    final output = <List<double>>[
      List<double>.filled(_labels.length, 0.0),
    ];

    interpreter.run(input, output);

    return List<double>.from(output[0]);
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

    // Use the last few frames (newest last); fall back to the current hand.
    final cutoff =
        DateTime.now().subtract(_historyWindow);

    final recent = _handHistory
        .where((entry) => entry.time.isAfter(cutoff))
        .map((entry) => entry.hand)
        .toList();

    final frames = <Hand>[
      if (recent.length > _historyFrames)
        ...recent.sublist(recent.length - _historyFrames)
      else
        ...recent,
    ];

    if (frames.isEmpty) {
      frames.add(handsForRecognition.first);
    }

    frames.removeWhere(
      (h) => h.landmarks.length != numLandmarks,
    );

    if (frames.isEmpty) {
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
      // STEP 1-3: FEATURES + MODEL, FOR EACH ORIENTATION, AVERAGED OVER FRAMES
      // ======================================================================

      debugPrint('');
      debugPrint('========================================');
      debugPrint('STARTING LANDMARK RECOGNITION');
      debugPrint('========================================');
      debugPrint('Frames used: ${frames.length}');

      final orientations = tryBothOrientations
          ? <bool>[false, true]
          : <bool>[false];

      List<double>? scores;
      bool usedMirror = false;
      double bestOverall = -1.0;

      for (final mirror in orientations) {
        final averaged =
            List<double>.filled(_labels.length, 0.0);

        for (final frame in frames) {
          final features = _landmarksToFeatures(
            frame,
            mirror: mirror,
          );

          if (identical(frame, frames.last)) {
            debugPrint(
              'FEATURES mirror=$mirror: '
              '${features.map((e) => e.toStringAsFixed(3)).toList()}',
            );
          }

          final frameScores =
              _runModel(interpreter, features);

          for (int i = 0; i < averaged.length; i++) {
            averaged[i] += frameScores[i] / frames.length;
          }
        }

        double peak = averaged[0];

        for (int i = 1; i < averaged.length; i++) {
          if (averaged[i] > peak) {
            peak = averaged[i];
          }
        }

        debugPrint(
          'ORIENTATION mirror=$mirror '
          'scores=${averaged.map((e) => e.toStringAsFixed(3)).toList()} '
          'peak=${peak.toStringAsFixed(3)}',
        );

        if (peak > bestOverall) {
          bestOverall = peak;
          scores = averaged;
          usedMirror = mirror;
        }
      }

      if (scores == null) {
        throw Exception('The model returned no scores.');
      }

      debugPrint('Chosen orientation: mirror=$usedMirror');
      debugPrint('');
      debugPrint('PREDICTION SCORES:');

      for (int i = 0; i < scores.length && i < _labels.length; i++) {
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

      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];

          bestIndex = i;
        }
      }

      debugPrint('');
      debugPrint('BEST LABEL: ${_labels[bestIndex]}');
      debugPrint(
        'BEST CONFIDENCE: '
        '${(bestScore * 100).toStringAsFixed(2)}%',
      );
      debugPrint('========================================');
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
              'Closest match: ${_friendlyLabel(_labels[bestIndex])} '
              '(${(bestScore * 100).toStringAsFixed(0)}%). '
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

      case 'pointing':
        return '👈 Pointing';

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

// ============================================================================
// TIMESTAMPED HAND (for averaging recent frames on capture)
// ============================================================================

class _TimedHand {
  _TimedHand(this.hand, this.time);

  final Hand hand;

  final DateTime time;
}