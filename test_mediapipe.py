from mediapipe.tasks import python as mp_python

# Check what's available
vision = mp_python.vision
print("Vision module attributes:")
print([x for x in dir(vision) if not x.startswith('_')])

# Try to access HolisticLandmarkerOptions
try:
    options_class = vision.HolisticLandmarkerOptions
    print(f"\nHolisticLandmarkerOptions found: {options_class}")
    print(f"Signature: {options_class.__init__.__annotations__}")
except AttributeError as e:
    print(f"\nError accessing HolisticLandmarkerOptions: {e}")

# Try to access RunningMode
try:
    running_mode = vision.RunningMode
    print(f"\nRunningMode found: {running_mode}")
except AttributeError as e:
    print(f"\nError accessing RunningMode: {e}")

# Try to access HolisticLandmarker
try:
    holistic = vision.HolisticLandmarker  
    print(f"\nHolisticLandmarker found: {holistic}")
except AttributeError as e:
    print(f"\nError accessing HolisticLandmarker: {e}")

