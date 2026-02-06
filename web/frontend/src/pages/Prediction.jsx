import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { submitPrediction, fetchCountries } from '../api/client';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

export default function Prediction() {
  const [selectedCountry, setSelectedCountry] = useState('Algeria');
  const [inputs, setInputs] = useState({
    ozone: 300,
    cloud: 50,
    aerosol: 1.5,
    temp: 25,
    time: 12
  });

  const { data: countriesData } = useQuery({
    queryKey: ['countries'],
    queryFn: fetchCountries
  });

  const mutation = useMutation({
    mutationFn: submitPrediction
  });

  const handleSliderChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    mutation.mutate({ 
      country: selectedCountry,
      ozone: inputs.ozone,
      cloud_cover: inputs.cloud,
      aerosol: inputs.aerosol,
      temperature: inputs.temp,
      hour: inputs.time,
      timestamp: new Date().toISOString()
    });
  };

  const sliderConfig = [
    { name: 'ozone', label: 'Ozone Layer', unit: 'DU', min: 200, max: 500, step: 5, color: 'blue', hint: 'UV absorption shield' },
    { name: 'cloud', label: 'Cloud Cover', unit: '%', min: 0, max: 100, step: 1, color: 'slate', hint: 'Surface UV reduction' },
    { name: 'aerosol', label: 'Aerosol Index', unit: '', min: 0, max: 5, step: 0.1, color: 'amber', hint: 'Particle scattering' },
    { name: 'temp', label: 'Temperature', unit: '°C', min: -10, max: 50, step: 1, color: 'red', hint: 'Atmospheric density' },
    { name: 'time', label: 'Solar Hour', unit: ':00', min: 6, max: 20, step: 1, color: 'yellow', hint: 'Peak at noon' },
  ];

  const getColorClasses = (color) => ({
    blue: { label: 'text-blue-300', value: 'text-blue-400', accent: 'accent-blue-500', track: 'from-blue-500/20 to-blue-500', glow: 'shadow-blue-500/20' },
    slate: { label: 'text-slate-300', value: 'text-slate-400', accent: 'accent-slate-400', track: 'from-slate-500/20 to-slate-400', glow: 'shadow-slate-400/20' },
    amber: { label: 'text-amber-300', value: 'text-amber-400', accent: 'accent-amber-500', track: 'from-amber-500/20 to-amber-500', glow: 'shadow-amber-500/20' },
    red: { label: 'text-red-300', value: 'text-red-400', accent: 'accent-red-500', track: 'from-red-500/20 to-red-500', glow: 'shadow-red-500/20' },
    yellow: { label: 'text-yellow-300', value: 'text-yellow-400', accent: 'accent-yellow-500', track: 'from-yellow-500/20 to-yellow-500', glow: 'shadow-yellow-500/20' },
  }[color]);

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-violet-100 to-white/50 inline-block mb-1">
            UV Prediction Engine
          </h2>
          <p className="text-slate-400 text-sm font-medium tracking-wide">
            Simulate radiation levels with atmospheric parameters
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

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
        
        {/* Control Panel - Takes 3 columns */}
        <div className="lg:col-span-3">
          <div className="glass-panel rounded-2xl p-8 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-violet-500 via-blue-500 to-purple-500 opacity-50"></div>
            
            <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-8 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-violet-400 shadow-[0_0_8px_rgba(167,139,250,0.8)]"></span>
              Atmospheric Control Matrix
            </h3>
            
            <form onSubmit={handleSubmit} className="space-y-8">
              {sliderConfig.map(slider => {
                const colors = getColorClasses(slider.color);
                const value = inputs[slider.name];
                const percent = ((value - slider.min) / (slider.max - slider.min)) * 100;
                
                return (
                  <div key={slider.name} className="group">
                    <div className="flex justify-between items-center mb-3">
                      <div>
                        <label className={`text-sm font-semibold ${colors.label} tracking-wide`}>
                          {slider.label}
                        </label>
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">{slider.hint}</p>
                      </div>
                      <div className={`font-mono text-lg font-bold ${colors.value} tabular-nums`}>
                        {value}{slider.unit}
                      </div>
                    </div>
                    
                    <div className="relative h-3 bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className={`absolute left-0 top-0 h-full bg-gradient-to-r ${colors.track} transition-all duration-150 pointer-events-none`}
                        style={{ width: `${percent}%` }}
                      ></div>
                      <input 
                        type="range" 
                        name={slider.name} 
                        min={slider.min} 
                        max={slider.max} 
                        step={slider.step} 
                        value={value} 
                        onChange={handleSliderChange}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                    </div>
                  </div>
                );
              })}

              <button 
                type="submit" 
                disabled={mutation.isPending}
                className="w-full relative group mt-6"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-600 to-indigo-600 rounded-xl blur opacity-60 group-hover:opacity-100 transition duration-200"></div>
                <div className="relative px-8 py-4 bg-[#0a0a0f] rounded-xl leading-none flex items-center justify-center gap-3 text-white font-bold tracking-wide">
                  {mutation.isPending ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Computing...</span>
                    </>
                  ) : (
                    <>
                      <span>Analyze Radiation</span>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </>
                  )}
                </div>
              </button>
            </form>
          </div>
        </div>

        {/* Result Display - Takes 2 columns */}
        <div className="lg:col-span-2 flex items-center justify-center min-h-[500px]">
          {mutation.isIdle && (
            <div className="text-center">
              <div className="relative">
                <div className="w-48 h-48 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center mx-auto animate-pulse">
                  <div className="w-32 h-32 rounded-full border border-white/5 flex items-center justify-center">
                    <span className="text-slate-500 text-sm font-medium uppercase tracking-widest">Standby</span>
                  </div>
                </div>
              </div>
              <p className="text-slate-500 mt-6 text-sm">Configure parameters to begin analysis</p>
            </div>
          )}

          {mutation.isSuccess && (
            <div className="relative group">
              <div className={`absolute -inset-8 rounded-full blur-2xl transition duration-500 animate-pulse ${
                mutation.data.prediction > 8 ? 'bg-red-600/30' : 
                mutation.data.prediction > 5 ? 'bg-orange-600/30' :
                mutation.data.prediction > 3 ? 'bg-yellow-600/30' : 'bg-green-600/30'
              }`}></div>
              
              <div className="relative w-72 h-72 rounded-full flex flex-col items-center justify-center glass-panel border-2 border-white/10">
                <div className={`absolute inset-2 rounded-full border ${
                  mutation.data.prediction > 8 ? 'border-red-500/30' : 
                  mutation.data.prediction > 5 ? 'border-orange-500/30' :
                  mutation.data.prediction > 3 ? 'border-yellow-500/30' : 'border-green-500/30'
                }`}></div>
                
                <span className="text-slate-500 text-[10px] uppercase tracking-[0.3em] mb-1">Predicted UV</span>
                <span className="text-slate-400 text-xs mb-3">{selectedCountry}</span>
                
                <span className={`text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b ${
                  mutation.data.prediction > 8 ? 'from-red-300 to-red-600' : 
                  mutation.data.prediction > 5 ? 'from-orange-300 to-orange-600' :
                  mutation.data.prediction > 3 ? 'from-yellow-300 to-yellow-600' : 'from-green-300 to-green-600'
                }`} style={{ textShadow: `0 0 40px ${
                  mutation.data.prediction > 8 ? 'rgba(239, 68, 68, 0.5)' : 
                  mutation.data.prediction > 5 ? 'rgba(251, 146, 60, 0.5)' :
                  mutation.data.prediction > 3 ? 'rgba(250, 204, 21, 0.5)' : 'rgba(34, 197, 94, 0.5)'
                }` }}>
                  {mutation.data.prediction?.toFixed(1) || '0.0'}
                </span>
                
                <span className={`mt-4 px-4 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest border ${
                  mutation.data.prediction > 8 ? 'bg-red-500/10 text-red-400 border-red-500/30' : 
                  mutation.data.prediction > 5 ? 'bg-orange-500/10 text-orange-400 border-orange-500/30' :
                  mutation.data.prediction > 3 ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30' : 'bg-green-500/10 text-green-400 border-green-500/30'
                }`}>
                  {mutation.data.risk_level || (mutation.data.prediction > 8 ? 'Extreme' : 
                   mutation.data.prediction > 5 ? 'High' :
                   mutation.data.prediction > 3 ? 'Moderate' : 'Low')}
                </span>
              </div>
            </div>
          )}

          {mutation.isError && (
            <div className="glass-panel rounded-2xl p-8 text-center border-red-500/20 max-w-sm">
              <div className="w-14 h-14 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4 border border-red-500/20">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <p className="font-bold text-white">Analysis Failed</p>
              <p className="mt-2 text-sm text-red-200/60">Backend connection interrupted</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
