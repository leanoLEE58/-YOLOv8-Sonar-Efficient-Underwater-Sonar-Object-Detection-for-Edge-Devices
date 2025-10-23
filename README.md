# 🤖 YOLOv8-Sonar: Efficient Underwater Sonar Object Detection for Edge Devices

<div align="center">

[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Enhanced-brightgreen.svg)]()
[![Edge AI](https://img.shields.io/badge/Edge-Deployment-blue.svg)]()


**Lightweight Sonar Object Detection for Underwater Robotics**

*Ocean University of China - Underwater Robotics Lab*

🚧 **Project Status: Active Development** 🚧

</div>

---

## 📋 Overview

This project develops an **improved YOLOv8-based object detection system** specifically designed for **underwater sonar imaging** on **resource-constrained edge devices** (AUVs, ROVs, and underwater robots).

### 🎯 Key Challenges Addressed

- 🌊 **Sonar Image Characteristics**: Low resolution, speckle noise, limited contrast
- 🤖 **Edge Deployment**: Limited computational power on underwater robots
- 📊 **Data Scarcity**: Limited labeled underwater sonar datasets
- ⚡ **Real-time Requirements**: Fast inference for autonomous navigation

<div align="center">
<img src="assets/sonar_detection_demo.gif" alt="Detection Demo" width="600"/>

*Real-time sonar object detection on edge device*
</div>

---

## ✨ Key Features

### 🔬 Technical Innovations

- ✅ **Transfer Learning Strategy**: Pre-trained on optical underwater images, fine-tuned on sonar data
- ✅ **Lightweight Architecture**: Optimized YOLOv8 backbone for edge deployment
- ✅ **Sonar-Specific Augmentation**: Custom data augmentation for sonar characteristics
- ✅ **Multi-Scale Detection**: Enhanced feature pyramid for small target detection
- ✅ **Edge Optimization**: INT8 quantization and TensorRT acceleration

### 🎯 Target Applications

- 🐟 Marine life detection and tracking
- 🪨 Underwater obstacle avoidance
- 🔍 Subsea infrastructure inspection
- 🗺️ Autonomous underwater navigation
- ⚓ Wreck and artifact detection

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Sonar Image                        │
│                      (320×320 / 640×640)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Lightweight YOLOv8 Backbone                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CSPLayer   │→ │  C2f Module  │→ │   SPPF      │     │
│  │  (Reduced)   │  │  (Optimized) │  │  (Enhanced)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Sonar-Adapted Feature Pyramid                    │
│         (Multi-scale feature fusion)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Detection Head (Modified)                      │
│    Classification + Localization + Confidence               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Post-Processing (Optimized NMS)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
                 Detection Results


## 🗺️ Roadmap

### ✅ Completed
- [x] YOLOv8 baseline implementation
- [x] Transfer learning framework
- [x] Sonar data augmentation pipeline
- [x] Initial model training and validation

### 🚧 In Progress
- [ ] Advanced sonar-specific preprocessing
- [ ] Attention mechanism integration
- [ ] Multi-task learning (detection + segmentation)
- [ ] Extensive edge device testing

### 📅 Planned
- [ ] Real-time tracking integration
- [ ] 3D bounding box estimation
- [ ] Acoustic-optical sensor fusion
- [ ] Larger-scale dataset collection
- [ ] Benchmark on public sonar datasets
- [ ] Paper submission

---

## 📊 Evaluation

```bash
# Evaluate on test set
python val.py --weights weights/best.pt \
              --data configs/sonar.yaml \
              --img-size 640 \
              --batch-size 8

# Generate confusion matrix
python val.py --weights weights/best.pt \
              --data configs/sonar.yaml \
              --task test \
              --save-json \
              --save-confusion-matrix

# Benchmark on edge device
python benchmark.py --weights weights/best-int8.trt \
                    --device jetson-nano \
                    --iterations 100
```

---

## 🤝 Contributing

This is an active research project. We welcome:

- 🐛 Bug reports and fixes
- 💡 Feature suggestions
- 📊 Dataset contributions
- 📖 Documentation improvements

Please open an issue or submit a pull request!

---

## 📚 References

### Key Technologies

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Transfer Learning**: Domain adaptation for sonar imaging
- **TensorRT**: NVIDIA deep learning optimization SDK
- **Edge AI**: Deployment on resource-constrained devices

### Related Publications

```bibtex
@article{jocher2023ultralytics,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023}
}

@inproceedings{li2025uwis,
  title={UWIS: Underwater Image Stitching Dataset and Pipeline},
  author={Li, Jiayi and Dong, Kaizhi and Li, Guihui and others},
  booktitle={OCEANS 2025},
  year={2025}
}
```

---

## 📧 Contact

**Project Lead:**
- **Jiayi Li** - jiayilee@stu.ouc.edu.cn
- **Kaizhi Dong** - dongkaizhi@stu.ouc.edu.cn

**Supervisor:**
- **rongshenghui** - rongshenghui@ouc.edu.cn

**Institution:**  
Ocean University of China  
Qingdao, Shandong, China

---

## 📜 License（pending）

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Ultralytics team for YOLOv8 framework
- Marine robotics community for dataset contributions
- NVIDIA for edge AI development tools
- Ocean University of China for research support

---

<div align="center">

### 🤖 Built for Underwater Robotics, Optimized for the Edge

**Questions or Collaboration? [Open an Issue](https://github.com/your-username/yolov8-sonar/issues)**

*Last Updated: Oct. 2025*

</div>
