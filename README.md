# Plane Cam
Plane Cam is a motorised camera module which extends the functionality of [PlaneTracker](https://github.com/PrzemekWasinski/PlaneTracker). When an aircraft is close enough Plane Tracker will send its latitude, 
latitude and altitude to PlaneCam, PlaneCam will then move the camera on the motorised pan-tilt mount to aim at the aircraft and take a picture. 

The picture taken by the Raspberry Pi Camera will then be analysed with OpenCV to check whether a plane has been captured, after this the picture and image recognition results will be sent back to PlaneTracker. 

![20260328_141810](https://github.com/user-attachments/assets/71c0b611-4652-42cd-b481-05b2729e411a)

Here is PlaneCam on my roof tracking a plane.

# Example Image
![planecam-capture](https://github.com/user-attachments/assets/3f739359-ba25-4a1f-aa43-c653efc795ff)

Here is an example of an image taken by PlaneCam automatically.

# Tech Stack:
```
GPIO Control & Calculations: C++
Image Recognition: OpenC
```

# Hardware:
```
Computer: Raspberry Pi 4
Motors: MG996R Servo Motors
Camera: Raspberry Pi High Quality Camera
Lens: Arducam 8-50mm Telephoto Lens
```
