import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchForecast, fetchCountries } from '../api/client';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import Card from '../components/ui/Card';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function Dashboard() {
  const [selectedCountry, setSelectedCountry] = useState('Algeria');
  
  const { data: countriesData } = useQuery({
    queryKey: ['countries'],
    queryFn: fetchCountries
  });
  
  const { data: forecastData, isLoading, error } = useQuery({
    queryKey: ['forecast', selectedCountry],
    queryFn: () => fetchForecast(selectedCountry)
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-96">
      <div className="text-center relative">
        <div className="absolute inset-0 bg-purple-500/20 blur-xl rounded-full"></div>
        <div className="w-20 h-20 border-4 border-t-purple-500 border-r-blue-500 border-b-purple-500 border-l-blue-500 rounded-full animate-spin mx-auto mb-6 relative z-10"></div>
        <p className="text-blue-200 text-lg font-medium tracking-widest uppercase animate-pulse">Scanning Atmosphere...</p>
      </div>
    </div>
  );
  
  if (error) return (
    <div className="text-center text-red-400 mt-20 glass-panel p-10 max-w-lg mx-auto border-red-500/30">
      <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      </div>
      <p className="text-xl font-bold text-white">Connection Interrupted</p>
      <p className="mt-2 text-red-200/60 text-sm">Satellite link failed. Check backend connectivity.</p>
    </div>
  );

  // Chart configuration with Neon Glow effects
  const peakUvChartData = {
    labels: forecastData?.forecast?.map(d => d.date.split('-').slice(1).join('/')) || [],
    datasets: [
      {
        label: 'Peak UV Index',
        data: forecastData?.forecast?.map(d => d.uv_max) || [],
        borderColor: '#f472b6', // Pink-400
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(0, 'rgba(244, 114, 182, 0.5)'); // Pink-400 glow
          gradient.addColorStop(1, 'rgba(244, 114, 182, 0)');
          return gradient;
        },
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#000',
        pointBorderColor: '#f472b6',
        pointBorderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointHoverBackgroundColor: '#f472b6',
        pointHoverBorderColor: '#fff',
        borderWidth: 3,
      },
      {
        label: 'Temperature (°C)',
        data: forecastData?.forecast?.map(d => d.temp_c) || [],
        borderColor: '#fb923c', // Orange-400
        backgroundColor: 'transparent',
        tension: 0.4,
        fill: false,
        borderDash: [5, 5],
        pointRadius: 4,
        pointBackgroundColor: '#fb923c',
        pointBorderColor: 'transparent',
        borderWidth: 2,
        yAxisID: 'y1'
      }
    ]
  };

  const seasonalComparisonData = {
    labels: forecastData?.monthly_averages?.map(m => m.month_name) || [],
    datasets: [
      {
        label: 'Average UV (Model Output)',
        data: forecastData?.monthly_averages?.map(m => m.uv_avg) || [],
        borderColor: '#a78bfa', // Violet-400
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 250);
          gradient.addColorStop(0, 'rgba(167, 139, 250, 0.4)');
          gradient.addColorStop(1, 'rgba(167, 139, 250, 0)');
          return gradient;
        },
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#000',
        pointBorderColor: '#a78bfa',
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        borderWidth: 3,
      }
    ]
  };

  const monthlyChartData = {
    labels: forecastData?.monthly_averages?.map(m => m.month_name) || [],
    datasets: [
      {
        label: 'Monthly Peak UV',
        data: forecastData?.monthly_averages?.map(m => m.uv_peak) || [],
        backgroundColor: forecastData?.monthly_averages?.map(m => {
          const uv = m.uv_peak;
          if (uv < 3) return 'rgba(34, 197, 94, 0.8)';
          if (uv < 6) return 'rgba(250, 204, 21, 0.8)';
          if (uv < 8) return 'rgba(251, 146, 60, 0.8)';
          return 'rgba(248, 113, 113, 0.8)';
        }) || [],
        borderRadius: 8,
        borderWidth: 0,
        hoverBackgroundColor: '#fff',
      }
    ]
  };

  const yearlyChartData = {
    labels: forecastData?.yearly_trend?.map(y => y.year) || [],
    datasets: [
      {
        label: 'Yearly Avg Peak UV',
        data: forecastData?.yearly_trend?.map(y => y.avg_uv) || [],
        borderColor: '#38bdf8', // Sky-400
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 250);
          gradient.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
          gradient.addColorStop(1, 'rgba(56, 189, 248, 0)');
          return gradient;
        },
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#000',
        pointBorderColor: '#38bdf8',
        pointBorderWidth: 2,
        pointRadius: 5,
        borderWidth: 3,
      }
    ]
  };

  const baseChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: { 
          color: '#94a3b8', 
          font: { family: 'sans-serif', size: 11, weight: 600 },
          usePointStyle: true,
          boxWidth: 8
        }
      },
      tooltip: {
        backgroundColor: 'rgba(10, 10, 15, 0.9)',
        titleColor: '#fff',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        displayColors: true,
        boxPadding: 4
      }
    },
    scales: {
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.03)' },
        ticks: { color: '#64748b', font: { size: 10 } },
        border: { display: false }
      },
      x: {
        grid: { display: false },
        ticks: { color: '#64748b', font: { size: 10 } },
        border: { display: false }
      }
    }
  };

  const peakChartOptions = {
    ...baseChartOptions,
    scales: {
      y: {
        ...baseChartOptions.scales.y,
        suggestedMax: 12,
        title: { display: true, text: 'Peak UV Index', color: '#64748b', font: { size: 10 } }
      },
      y1: {
        position: 'right',
        grid: { display: false },
        ticks: { color: '#fb923c' },
        title: { display: true, text: 'Temp (°C)', color: '#fb923c', font: { size: 10 } },
        border: { display: false }
      },
      x: baseChartOptions.scales.x
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-blue-100 to-white/50 inline-block mb-1">
            Global Radiation Monitor
          </h2>
          <p className="text-slate-400 text-sm font-medium tracking-wide">
            Real-time atmospheric analysis & predictive modeling
          </p>
        </div>
        
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-violet-600 to-indigo-600 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-200"></div>
          <div className="relative flex items-center bg-[#0a0a0f] border border-white/10 rounded-lg p-1 pr-4">
            <span className="px-3 text-indigo-400 text-xs font-bold uppercase tracking-wider">Region</span>
            <div className="h-4 w-px bg-white/10 mx-1"></div>
            <select 
              value={selectedCountry}
              onChange={(e) => setSelectedCountry(e.target.value)}
              className="bg-transparent text-white text-sm font-medium focus:outline-none cursor-pointer py-1.5"
            >
              {countriesData?.countries?.map(country => (
                <option key={country} value={country} className="bg-slate-900">{country}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Hero Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-50 group-hover:opacity-100 transition-opacity">
            <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
              <span className="text-blue-400">📍</span>
            </div>
          </div>
          <h3 className="text-xs font-bold text-blue-300 uppercase tracking-widest mb-1">Target Location</h3>
          <p className="text-2xl font-bold text-white tracking-tight">{forecastData.location}</p>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-transparent opacity-50"></div>
        </div>
        
        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute inset-0 bg-red-500/5 group-hover:bg-red-500/10 transition-colors duration-500"></div>
          <h3 className="text-xs font-bold text-red-300 uppercase tracking-widest mb-1">Peak UV Today</h3>
          <div className="flex items-end gap-3">
            <p className="text-4xl font-bold text-white glow-text" style={{ textShadow: forecastData.current_uv > 8 ? '0 0 20px rgba(239, 68, 68, 0.5)' : 'none' }}>
              {forecastData.current_uv}
            </p>
            <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide mb-1.5 ${
              forecastData.risk_level === 'Low' ? 'bg-green-500/20 text-green-300 border border-green-500/30' :
              forecastData.risk_level === 'Moderate' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30' :
              forecastData.risk_level === 'High' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' :
              'bg-red-500/20 text-red-300 border border-red-500/30'
            }`}>
              {forecastData.risk_level}
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <h3 className="text-xs font-bold text-orange-300 uppercase tracking-widest mb-1">Temperature</h3>
          <p className="text-3xl font-bold text-white mt-1">{forecastData.current_temp}°C</p>
          <div className="w-full bg-white/10 h-1.5 rounded-full mt-4 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-orange-400 to-red-500" 
              style={{ width: `${Math.min(100, Math.max(0, (forecastData.current_temp + 10) * 1.6))}%` }}
            ></div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <h3 className="text-xs font-bold text-emerald-300 uppercase tracking-widest mb-1">Condition</h3>
          <p className="text-xl font-bold text-white mt-2 leading-tight">{forecastData.forecast?.[0]?.condition}</p>
        </div>
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[400px]">
        <Card title="7-Day Peak UV Forecast" className="lg:col-span-2 h-full">
          <div className="h-full w-full pb-4">
            <Line data={peakUvChartData} options={peakChartOptions} />
          </div>
        </Card>
        <Card title="Seasonal Pattern" className="h-full">
          <div className="h-full w-full pb-4">
            <Line data={seasonalComparisonData} options={baseChartOptions} />
          </div>
        </Card>
      </div>

      {/* Secondary Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Monthly Peak Analysis" className="h-[280px]">
          <div className="h-full w-full pb-4">
            {forecastData?.monthly_averages?.length > 0 ? (
              <Bar data={monthlyChartData} options={baseChartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-slate-400 text-sm">No data available</div>
            )}
          </div>
        </Card>

        <Card title="Historical Trend (15 years)" className="h-[280px]">
          <div className="h-full w-full pb-4">
            {forecastData?.yearly_trend?.length > 0 ? (
              <Line data={yearlyChartData} options={baseChartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full text-slate-400 text-sm">No data available</div>
            )}
          </div>
        </Card>
      </div>

      {/* Modern Legend */}
      <div className="glass-panel rounded-xl p-6">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">UV Index Risk Spectrum</h4>
        <div className="relative h-4 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-purple-600 mb-2"></div>
        <div className="flex justify-between text-[10px] font-medium text-slate-400 uppercase tracking-wider">
          <span>Low (0-2)</span>
          <span>Moderate (3-5)</span>
          <span>High (6-7)</span>
          <span>Very High (8-10)</span>
          <span>Extreme (11+)</span>
        </div>
      </div>
    </div>
  );
}
