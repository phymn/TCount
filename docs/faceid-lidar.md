# How does Face ID work

Apple’s Face ID recognizes you using **3D** facial data—not just a selfie. It measures the contours of your face with invisible infrared light, then compares the result with the facial data securely stored on your iPhone or iPad Pro.

## What happens when you look at your device

1. The TrueDepth system illuminates your face. An infrared flood illuminator helps the system see you even in darkness. A dot projector projects thousands of invisible infrared dots onto your face.
2. An infrared camera reads the pattern. The system uses those dots to build a depth map—capturing the shape of features such as your nose, cheeks, and eye sockets—and also captures an infrared image. This gives it information about facial geometry rather than just color and appearance.
3. Neural processing turns the scan into a mathematical representation. Instead of simply comparing two photographs, Face ID compares that representation with the one created when you enrolled your face. The matching takes place within the protection of Apple’s Secure Enclave, the device’s security subsystem.
4. A successful match authorizes access. Face ID can unlock the device, approve a purchase, or authenticate you in a supported app. By default, it also checks that your eyes are open and you’re looking toward the device; that attention requirement can be disabled for accessibility.

## Why a photo usually won’t fool it

A photograph can resemble your face but does not have your face’s depth. Face ID combines depth information with neural-network-based anti-spoofing checks, rather than relying on a normal camera image alone.

It also adapts to changes such as makeup or growing facial hair. After a major change—such as shaving off a full beard—it may ask for your passcode before updating its stored facial data.

## Where your face data goes

The enrolled facial representation stays protected on your device; it is not sent to Apple’s servers. Think of Face ID as a local biometric verification system: the device checks whether the person in front of it matches the enrolled user, rather than searching an online database to identify a face.

---

# How does LiDAR on an iPhone work?

The iPhone’s LiDAR scanner measures distance by sending out invisible infrared laser pulses and timing how long their reflections take to return. It uses those measurements to build a **depth** map of nearby objects and surfaces—not just a flat photograph. LiDAR stands for “Light Detection and Ranging.”

## How it works, step by step

1. Emit infrared light. The rear-facing scanner sends a pattern of infrared light pulses into the scene. You can’t see these pulses with your eyes.
2. Detect the reflections. A sensor receives light bouncing back from walls, furniture, people, and other surfaces. Nearby surfaces return the light sooner than distant ones.
3. Calculate distance. This is called *time of flight*. The relationship is:

$$
\text{Distance}=\frac{\text{speed of light}\times\text{round-trip travel time}}{2}
$$

The division by two accounts for the light traveling to the object and back. For a surface 5 meters away, that round trip takes only about 33 nanoseconds.
4. Build a spatial map. The phone combines depth readings with camera images and motion-sensor data to understand surfaces and its position as you move. Software can use that information to construct a 3D representation of a room or object.

Unlike a large rotating laser scanner, the iPhone’s LiDAR has no moving scanning parts. Its useful range is roughly 5 meters, or 16 feet.

## What the iPhone uses it for

- Augmented reality: Place virtual objects on real floors or tables, with better awareness of the surrounding space.
- Measurements and 3D scanning: Estimate dimensions and create room or object models in compatible apps.
- Camera focusing: Provide distance information that helps the camera focus, particularly in low light.

It is useful for room-scale mapping, but it is not a substitute for precision surveying equipment. Range, surface properties, and lighting affect the results.

## How it differs from Face ID

Both use invisible infrared light, but they extract depth differently:

| System | Main method | Designed for |
| :-- | :-- | :-- |
| Face ID’s TrueDepth | Analyzes a projected infrared dot pattern to map facial geometry | Recognizing a face at close range |
| Rear LiDAR | Measures the return time of reflected infrared pulses | Measuring the surrounding scene out to several meters |

So the rear LiDAR is a scene-distance sensor, not the sensor that authenticates your face.

---

# Who emits those infrared lights?

A tiny laser-emitter chip inside the iPhone’s rear LiDAR module emits the infrared light—not the camera lenses or the camera flash. It uses an array of **VCSELs**, short for *vertical-cavity surface-emitting lasers*. These are microscopic semiconductor lasers built together on a chip.

The module has two separate jobs:

- Emitter: The VCSEL array generates infrared pulses, and an optical element spreads them into a pattern of points across the scene.
- Receiver: A separate light-sensitive sensor called a *SPAD array*—single-photon avalanche diodes—detects the returning reflections so the system can measure their travel time.

The signal path is:

Electrical drive → infrared laser emitter → object → reflected light → receiver → distance calculation.

Both the emitter and receiver are part of the small LiDAR module beside the rear cameras. The regular cameras capture images; the LiDAR hardware supplies distance information.

---

# So the LiDAR module?

Yes—the **LiDAR module**. Its laser-emitter chip sends out the infrared pulses, and its receiver detects the light bouncing back. The module contains both parts.

---

# How many infrared lights emit LiDAR?

It depends on the iPhone model—and whether you mean the laser emitters inside the module or the infrared dots projected onto objects.

For the iPhone 12 Pro and 13 Pro, the LiDAR system uses:

- 64 microscopic infrared laser emitters on a chip.
- An optical element that splits their light into a pattern totaling **576** infrared dots. So, 576 dots does not mean 576 separate lasers.

Those numbers aren’t universal across all iPhones. Apple does not publicly disclose the detailed emitter and dot counts for every model.

---

## Sources

- [Apple Support: About Face ID advanced technology](https://support.apple.com/en-us/102381)
- [Wikipedia: Face ID](https://en.wikipedia.org/wiki/Face_ID)
- [Apple Support: Use Face ID on your iPhone or iPad Pro](https://support.apple.com/en-us/108411)
- [Apple Face ID Security Guide (PDF)](https://www.apple.com/business-docs/FaceID_Security_Guide.pdf)
- [Tech Advisor: What is LiDAR and what does it do on the iPhone?](https://www.techadvisor.com/article/742818/what-is-lidar-and-what-does-it-do-on-the-iphone.html)
- [CNET: LiDAR is one of the iPhone/iPad’s coolest tricks](https://www.cnet.com/tech/mobile/lidar-is-one-of-the-iphone-ipad-coolest-tricks-its-only-getting-better/)
- [Engineering.com: How to use the laser scanner in your pocket](https://www.engineering.com/how-to-use-the-laser-scanner-in-your-pocket/)
- [Nature: research on VCSEL LiDAR](https://www.nature.com/articles/s41598-021-01763-9)
- [PMC: LiDAR module details](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537187/)
