## Parking Monitor
<p align="center">
	<img src="https://img.shields.io/github/last-commit/RayvelGB/ParkingMonitor?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/RayvelGB/ParkingMonitor?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/RayvelGB/ParkingMonitor?style=default&color=0080ff" alt="repo-language-count">
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
- [🔰 Contributing](#-contributing)

---

## 📍 Overview

<p>
	This project is a comprehensive, real-time vehicle detection and counting system designed to turn any standard RTSP camera feed into an intelligent monitoring tool. It provides two primary functionalities: a "Tripwire" mode for counting vehicles as they cross virtual lines and a "Parking" mode for monitoring the real-time occupancy of parking spaces. The system is designed for high performance and scalability, making it suitable for traffic analysis, parking management, and general security surveillance.
</p>
<p>
	Leveraging a powerful backend built with Python and FastAPI, the system's intelligence is driven by a sophisticated object detection model. The core of this AI is the Faster R-CNN (Region-based Convolutional Neural Network) architecture, implemented in PyTorch. This two-stage detection model is renowned for its high accuracy, making it ideal for reliably identifying and locating vehicles even in complex scenes. The system is managed through a web-based user interface that allows users to configure cameras, define detection zones, and view live, annotated video streams, with all data persistently stored in a MySQL database.
</p>
<p>
	The remarkable effectiveness of the AI model stems from its training on the specialized VisDrone dataset. This large-scale dataset, captured entirely by drones, provides the model with expert knowledge of how to identify vehicles from the high-angle and oblique perspectives typical of surveillance cameras. It contains thousands of images with small, dense, and partially occluded objects, training the model to excel in challenging real-world conditions where standard, eye-level datasets would fail. This specialized training is what allows the system to accurately detect cars, trucks, and buses from a typical RTSP feed.
</p>
<p>
	To ensure a smooth, stutter-free experience, the immense computational load of the AI model is isolated from the main web server. Each camera's entire video processing pipeline runs in a completely separate multiprocessing process. This architecture bypasses Python's Global Interpreter Lock (GIL), allowing the heavy AI inference to run on a dedicated CPU core in parallel. This leaves the FastAPI application fully responsive and capable of serving multiple, high-framerate video streams without lag or interruption.
</p>

---

## 👾 Features

- **Real-Time Detection**: Processes live RTSP streams with minimal latency.
- **Tripwire Mode**: Counts vehicles entering and exiting defined zones by crossing virtual lines.
- **Parking Mode**: Monitors the occupancy status (vacant/occupied) of individual parking spaces.
- **Web-Based UI**: Modern and intuitive interface for camera management, zone definition, and live stream viewing.
- **High-Performance Backend**: Built with FastAPI for asynchronous, high-throughput performance.
- **Advanced AI Model**: Utilizes a Faster R-CNN model trained on the VisDrone dataset for high-accuracy aerial vehicle detection.
- **Parallel Processing**: Each camera runs in a separate process to prevent stuttering and maximize CPU usage, bypassing Python's GIL.
- **Persistent Storage**: All configurations and event logs are stored in a MySQL database.

---

## 📁 Project Structure

```sh
└── ParkingMonitor/
    ├── db_configs.py
    ├── detector.py
    ├── main.py
    ├── models
    │   └── fasterrcnn_visdrone.pth
    ├── requirements.txt
    └── static
        ├── auth_user.html
        ├── favicon.png
        ├── picker.html
        ├── register_camera.html
        ├── set_spaces.html
        └── stream.html
```

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with ParkingMonitor, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


### ⚙️ Installation

Install ParkingMonitor using one of the following methods:

**Build from source:**

1. Clone the ParkingMonitor repository:
```sh
❯ git clone https://github.com/RayvelGB/ParkingMonitor
```

2. Navigate to the project directory:
```sh
❯ cd ParkingMonitor
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```

### 🤖 Usage
Run ParkingMonitor using the following command:
**Using `uvicorn`**

```sh
❯ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/RayvelGB/ParkingMonitor/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/RayvelGB/ParkingMonitor/issues)**: Submit bugs found or log feature requests for the `ParkingMonitor` project.
- **💡 [Submit Pull Requests](https://github.com/RayvelGB/ParkingMonitor/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/RayvelGB/ParkingMonitor
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/RayvelGB/ParkingMonitor/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=RayvelGB/ParkingMonitor">
   </a>
</p>
</details>

---
