import { useState } from 'react';
import LandingPage from './pages/LandingPage';
import ForecastDashboard from './pages/ForecastDashboard';

export default function App() {
  const [entered, setEntered] = useState(false);

  if (entered) {
    return <ForecastDashboard />;
  }

  return <LandingPage onEnter={() => setEntered(true)} />;
}