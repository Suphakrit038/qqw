# 🔮 Amulet-AI Production# 🔮 Amulet-AI Production



Production-ready version ของระบบจำแนกพระเครื่องด้วย AI**ระบบจำแนกพระเครื่องอัจฉริยะ - Production Ready**



## 📁 โครงสร้างโปรเจคThai Amulet Classification System - Optimized for Production Deployment



```## 🚀 Quick Start

amulet-ai-production/

├── ai_models/              # 🧠 AI Models & Classification### 1. Install Dependencies

│   ├── enhanced_production_system.py```bash

│   ├── updated_classifier.pypip install -r requirements.txt

│   ├── compatibility_loader.py```

│   ├── labels.json

│   └── twobranch/         # Two-Branch CNN System### 2. Start API Server

├── api/                   # ⚡ API Endpoints```bash

│   ├── main_api.py       # Main API servercd api

│   ├── main_api_fast.py  # Fast API serverpython -m uvicorn main_api:app --host 0.0.0.0 --port 8000

│   └── main_api_secure.py # Secure API server```

├── frontend/              # 🎨 Web Interface

│   ├── main_app.py       # Streamlit app### 3. Start Frontend

│   ├── components/       # UI components```bash

│   └── utils/           # Utilitiescd frontend

├── core/                  # 🔧 Core Systemstreamlit run main_app.py --server.port 8501

│   ├── config.py         # Configuration```

│   ├── error_handling.py # Error management

│   └── performance.py    # Performance monitoring### 4. Access Application

├── dataset/              # 📊 Dataset- **Web App**: http://localhost:8501

│   ├── metadata/         # Dataset information- **API**: http://localhost:8000

│   └── splits/          # Train/Val/Test images- **API Docs**: http://localhost:8000/docs

├── config/               # ⚙️ Configuration

└── deployment/           # 🚀 Docker & Deployment## 📁 Project Structure

```

```

## 🚀 Quick Startamulet-ai-production/

├── ai_models/              # AI Models & Classification Logic

### 1. ติดตั้ง Dependencies│   ├── __init__.py

│   ├── enhanced_production_system.py

```bash│   ├── updated_classifier.py

pip install -r requirements.txt│   ├── compatibility_loader.py

```│   ├── labels.json

│   └── twobranch/         # Two-Branch CNN System

### 2. เริ่มต้น API Server├── api/                   # FastAPI Backend

│   ├── main_api.py        # Main API

```bash│   ├── main_api_fast.py   # Fast API

cd api│   └── main_api_secure.py # Secure API

python main_api.py├── frontend/              # Streamlit Frontend

# หรือ│   ├── main_app.py        # Main Web App

uvicorn main_api:app --host 0.0.0.0 --port 8000│   ├── style.css          # Custom Styles

```│   ├── components/        # UI Components

│   └── utils/             # Utilities

### 3. เริ่มต้น Frontend├── core/                  # Core Utilities

│   ├── config.py          # Configuration

```bash│   ├── error_handling.py  # Error Management

cd frontend│   ├── performance.py     # Performance Tools

streamlit run main_app.py --server.port 8501│   └── security.py        # Security Features

```├── organized_dataset/     # Training Dataset

│   ├── splits/            # Train/Val/Test Splits

### 4. เข้าใช้งาน│   ├── metadata/          # Dataset Metadata

│   └── processed/         # Processed Images

- **Frontend**: http://localhost:8501├── config/                # Configuration Files

- **API**: http://localhost:8000├── deployment/            # Docker & Deployment

- **API Docs**: http://localhost:8000/docs└── requirements.txt       # Python Dependencies

```

## 🎯 คุณสมบัติ

## 🎯 Features

- ✅ **Web Interface**: Streamlit-based UI

- ✅ **REST API**: FastAPI backend- **AI Classification**: Advanced machine learning models for amulet recognition

- ✅ **AI Classification**: Multiple AI models- **Two-Branch CNN**: Deep learning architecture for enhanced accuracy

- ✅ **Image Processing**: Advanced preprocessing- **Web Interface**: User-friendly Streamlit dashboard

- ✅ **Performance Monitoring**: Real-time metrics- **REST API**: FastAPI backend with comprehensive documentation

- ✅ **Docker Support**: Container deployment- **Security**: Built-in authentication and rate limiting

- ✅ **Security**: Authentication & validation- **Performance**: Optimized for production workloads

- **Docker Ready**: Container support for easy deployment

## 📊 Dataset

## 🔧 Configuration

Dataset ประกอบด้วยรูปภาพพระเครื่อง 6 ประเภท:

- phra_sivali### Environment Variables

- portrait_back  Copy `config/config_template.env` to `.env` and configure:

- prok_bodhi_9_leaves

- somdej_pratanporn_buddhagavak```bash

- waek_man# API Configuration

- wat_nong_e_dukAPI_HOST=0.0.0.0

API_PORT=8000

## 🧠 AI Models

# Frontend Configuration

1. **Enhanced Production System**: RandomForest + OOD DetectionFRONTEND_PORT=8501

2. **Two-Branch CNN**: PyTorch-based deep learning

3. **Updated Classifier**: Latest optimized model# Security

SECRET_KEY=your-secret-key-here

## 🐳 Docker DeploymentALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501



```bash# Model Configuration

# DevelopmentMODEL_PATH=ai_models/

docker-compose -f deployment/docker-compose.dev.yml upDATASET_PATH=organized_dataset/

```

# Production  

docker-compose -f deployment/docker-compose.prod.yml up### Docker Deployment

```

**Development:**

## 📈 Performance```bash

docker-compose -f deployment/docker-compose.dev.yml up

- ⚡ **Fast Response**: < 1s prediction time```

- 🎯 **High Accuracy**: > 95% validation accuracy

- 💾 **Memory Efficient**: Optimized for production**Production:**

- 🔄 **Scalable**: Horizontal scaling support```bash

docker-compose -f deployment/docker-compose.prod.yml up

## 🔒 Security```



- 🛡️ **Input Validation**: File type & size checking## 📊 Dataset

- 🔐 **API Authentication**: JWT-based security

- 🚦 **Rate Limiting**: Request throttlingThe system includes a comprehensive dataset with:

- 📊 **Monitoring**: Error tracking & logging- **6 Amulet Classes**: phra_sivali, portrait_back, prok_bodhi_9_leaves, etc.

- **Train/Validation/Test Splits**: Organized for machine learning

## 📝 API Endpoints- **Metadata**: Detailed information about each image

- **Augmented Data**: Enhanced training samples

- `POST /predict` - Image classification

- `GET /health` - Health check## 🧠 AI Models

- `GET /models/status` - Model status

- `GET /metrics` - Performance metrics### Available Models:

1. **Enhanced Production System**: scikit-learn based classifier

## 🔧 Configuration2. **Two-Branch CNN**: PyTorch deep learning model

3. **Updated Classifier**: Latest optimized model

แก้ไขการตั้งค่าใน:

- `config/config_template.env`### Model Performance:

- `core/config.py`- High accuracy on test dataset

- Real-time inference capability

## 📋 Production Checklist- Production-ready optimization



- ✅ Remove development dependencies## 🛡️ Security Features

- ✅ Optimize model loading

- ✅ Configure logging- Authentication & Authorization

- ✅ Set up monitoring- Rate Limiting

- ✅ Configure secrets- Input Validation

- ✅ Set up backups- Error Handling

- ✅ Performance testing- Secure Headers

- ✅ Security testing

## 📈 Performance

---

- Optimized inference speed

**ขนาดโปรเจค**: ~3.9GB (ส่วนใหญ่เป็น dataset)- Memory-efficient processing

**อัปเดตล่าสุด**: October 2025- Caching mechanisms

**เวอร์ชัน**: 4.0.0 Production- Async operations
- Load balancing ready

## 🔍 API Endpoints

### Classification
- `POST /predict` - Classify amulet images
- `POST /batch-predict` - Batch classification
- `GET /models` - Available models info

### System
- `GET /health` - Health check
- `GET /metrics` - System metrics
- `GET /status` - Service status

## 🎨 Frontend Features

- Drag & drop image upload
- Real-time predictions
- Interactive charts
- Model comparison
- Results export

## 🚀 Deployment Options

### Local Development
```bash
# Terminal 1: API
cd api && python -m uvicorn main_api:app --reload

# Terminal 2: Frontend  
cd frontend && streamlit run main_app.py
```

### Production Deployment
- Docker containers
- Kubernetes support
- Cloud deployment ready
- CI/CD pipeline compatible

## 📞 Support

For issues and questions:
- Check API documentation at `/docs`
- Review error logs in console
- Ensure all dependencies are installed
- Verify dataset paths are correct

## 📝 License

Copyright © 2025 Amulet-AI Team. All rights reserved.

---

**Built with ❤️ for Thai Cultural Heritage Preservation**