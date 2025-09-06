# Soil Analysis & Crop Recommendation API

A Flask-based API that provides soil analysis and crop recommendations based on location using real-time data from multiple APIs.

## Features

- **Real-time soil data** from SoilGrids API (sand, silt, clay, pH, organic carbon)
- **Weather data** from Open-Meteo API (temperature, humidity, rainfall)
- **NPK estimation** using scientific correlations
- **Crop recommendations** with confidence scores and alternatives
- **RESTful API** with JSON responses

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)
```bash
cp .env.example .env
# Edit .env file if you need custom configuration
```

### 3. Train the Model
```bash
python train_models.py
```

### 4. Run the API
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Soil Analysis & Crop Recommendation
```bash
POST /analyze-soil
Content-Type: application/json

{
    "location": "Delhi, India"
}
```

**Example Response:**
```json
{
    "success": true,
    "location": "Delhi, India",
    "analysis": {
        "primary_recommendation": {
            "crop": "rice",
            "confidence": 85.2
        },
        "alternative_crops": [
            {"name": "maize", "confidence": 12.8},
            {"name": "cotton", "confidence": 2.0}
        ],
        "soil_analysis": {
            "ph": 6.8,
            "sand_percent": 35.0,
            "silt_percent": 40.0,
            "clay_percent": 25.0,
            "organic_carbon": 1.2,
            "cation_exchange_capacity": 18.5
        },
        "climate_data": {
            "temperature_celsius": 28.5,
            "annual_rainfall_mm": 650,
            "humidity_percent": 75.0
        },
        "nutrient_estimates": {
            "nitrogen_kg_ha": 45.0,
            "phosphorus_kg_ha": 35.0,
            "potassium_kg_ha": 28.0
        }
    }
}
```

## Data Sources

- **SoilGrids API**: Global soil data (no API key required)
- **Open-Meteo Archive API**: Historical weather data (no API key required)
- **OpenWeather API**: Optional additional weather data (requires API key)

## Model Information

- **Algorithm**: Random Forest Classifier with hyperparameter tuning
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Target**: Crop types (rice, maize, cotton, etc.)
- **Training Data**: Crop recommendation dataset with soil and climate parameters

## Testing with curl

```bash
# Health check
curl http://localhost:5000/health

# Soil analysis
curl -X POST http://localhost:5000/analyze-soil \
  -H "Content-Type: application/json" \
  -d '{"location": "Mumbai, India"}'
```

## Production Deployment

For production deployment, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Optimization Notes

- Removed unnecessary Node.js/React frontend code (reduced from 45 lakh lines)
- Eliminated redundant NPK prediction models
- Integrated real APIs for soil data instead of placeholder values
- Optimized model training with hyperparameter tuning
- Added comprehensive error handling and logging
- Streamlined API endpoints for better usability

## API Rate Limits

- SoilGrids: ~1000 requests per day (free tier)
- Open-Meteo: No strict limits for reasonable usage
- Consider implementing caching for production use

# CropWise - Smart Crop Recommendation System

A modern, AI-powered web application that provides intelligent crop recommendations based on soil analysis, weather patterns, and agricultural best practices.

![CropWise Banner](https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80)

## 🌾 Features

- **Smart Soil Analysis**: Interactive form with real-time soil parameter validation
- **AI-Powered Recommendations**: Machine learning-based crop suggestions with confidence scoring
- **Weather Integration**: Optional geolocation-based weather data integration
- **Visual Analytics**: Radar charts showing soil profile vs. optimal conditions
- **History Tracking**: Save and manage past recommendations
- **Responsive Design**: Fully optimized for mobile and desktop use
- **Modern UI/UX**: Clean, agricultural-themed design with smooth animations

## 🛠 Tech Stack

- **Frontend**: Next.js 14 with React 18
- **Styling**: Tailwind CSS with custom agricultural color palette
- **Charts**: Recharts for data visualization
- **Icons**: Heroicons for consistent iconography
- **TypeScript**: Full type safety throughout the application
- **Responsive Design**: Mobile-first approach with Tailwind utilities

## 📋 Prerequisites

- Node.js 18+ and npm/yarn
- Modern web browser with JavaScript enabled

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

2. **Run Development Server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

3. **Open Application**
   Navigate to [http://localhost:3000](http://localhost:3000)

4. **Build for Production**
   ```bash
   npm run build
   npm run start
   # or
   yarn build
   yarn start
   ```

## 📱 Application Flow

### 1. Landing Page (`/`)
- Hero section with value proposition
- Feature highlights
- User testimonials
- Call-to-action buttons

### 2. Input Form (`/input`)
- Interactive soil parameter inputs (N, P, K, pH, temperature, humidity, rainfall)
- Real-time validation with optimal range indicators
- Advanced options with geolocation integration
- Responsive sliders and number inputs

### 3. Results Page (`/results`)
- Primary crop recommendation with confidence meter
- Alternative crop suggestions
- Detailed reasoning for recommendations
- Radar chart comparing soil profile to optimal conditions
- Save functionality for future reference

### 4. Dashboard (`/dashboard`)
- History of all past analyses
- Search and filter capabilities
- Table and grid view modes
- Analytics and statistics

## 🎨 Design System

### Color Palette
- **Primary Green**: Nature-inspired greens (#22c55e, #16a34a, #15803d)
- **Sky Blue**: Weather-related blues (#0ea5e9, #0284c7, #0369a1)
- **Earth Brown**: Soil-inspired browns (#8b5a2b, #7c2d12)
- **Neutral**: Clean grays for text and backgrounds

### Typography
- **Font Family**: Inter (Google Fonts)
- **Headings**: Bold, large sizes for hierarchy
- **Body Text**: Medium weight, optimized for readability

### Components
- **Cards**: Rounded corners (border-radius: 1rem)
- **Buttons**: Prominent CTAs with hover effects
- **Forms**: Clean inputs with focus states
- **Charts**: Interactive visualizations with tooltips

## 📊 Data Structure

### Soil Data Interface
```typescript
interface SoilData {
  nitrogen: number      // 0-140 mg/kg
  phosphorus: number    // 0-145 mg/kg
  potassium: number     // 5-205 mg/kg
  temperature: number   // 8-45 °C
  humidity: number      // 14-100 %
  ph: number           // 3.5-10.0 pH
  rainfall: number     // 20-300 mm
  location?: string    // Optional location
  timestamp?: string   // ISO date string
}
```

### Recommendation Interface
```typescript
interface CropRecommendation {
  name: string
  confidence: number        // 0-100%
  image: string
  description: string
  suitabilityScore: number
  reasons: string[]
  warnings?: string[]
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env.local` file for any API keys or configuration:
```env
NEXT_PUBLIC_WEATHER_API_KEY=your_weather_api_key
NEXT_PUBLIC_ANALYTICS_ID=your_analytics_id
```

### Tailwind Configuration
The project includes custom Tailwind configuration with:
- Custom color palette
- Extended animations
- Agricultural-themed utilities

## 🚦 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build production bundle
- `npm run start` - Start production server
- `npm run lint` - Run ESLint checks

## 📱 Responsive Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

All components are designed mobile-first with progressive enhancement.

## 🧪 Testing

The application includes comprehensive form validation and error handling:
- Input range validation
- Required field checking
- Local storage management
- Loading states and error messages

## 🔮 Future Enhancements

- **Backend Integration**: Connect to real ML models and weather APIs
- **User Authentication**: User accounts and personalized recommendations
- **Crop Database**: Expanded crop information and growing guides
- **Market Data**: Integration with commodity pricing
- **Offline Support**: PWA capabilities for field use
- **Multi-language**: Internationalization support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unsplash**: High-quality agricultural photography
- **Heroicons**: Beautiful iconography
- **Tailwind CSS**: Utility-first CSS framework
- **Next.js Team**: React framework and development tools
- **Recharts**: Composable charting library

---

Made with 🌱 by the CropWise team for modern farmers worldwide.
