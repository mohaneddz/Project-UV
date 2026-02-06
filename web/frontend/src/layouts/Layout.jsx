import React from 'react';
import { NavLink } from 'react-router-dom';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen text-white relative overflow-hidden">
      {/* Ambient background glow effects */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-purple-600/20 blur-[120px] pointer-events-none"></div>
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/20 blur-[120px] pointer-events-none"></div>

      <div className="relative z-10 flex flex-col min-h-screen">
        <header className="border-b border-white/5 bg-[#0a0a0f]/70 backdrop-blur-xl sticky top-0 z-50">
          <div className="container mx-auto px-6 h-20 flex items-center justify-between">
            <div className="flex items-center">
              {/* <h1 className="text-xl font-semibold text-white tracking-tight">
                LumiereGuard
              </h1> */}
            </div>

            <nav className="flex items-center gap-2 bg-white/5 p-1 rounded-xl border border-white/5 backdrop-blur-sm">
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  `px-5 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                    isActive 
                      ? 'bg-white/10 text-white shadow-inner' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`
                }
              >
                Dashboard
              </NavLink>
              <NavLink 
                to="/prediction" 
                className={({ isActive }) => 
                  `px-5 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                    isActive 
                      ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-lg shadow-indigo-500/25' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`
                }
              >
                Predict UV
              </NavLink>
            </nav>
          </div>
        </header>

        <main className="flex-1 container mx-auto px-6 py-10">
          {children}
        </main>

        {/* <footer className="border-t border-white/5 py-8 text-center bg-[#0a0a0f]/50 backdrop-blur-sm">
          <p className="text-gray-500 text-sm">© 2024 LumiereGuard AI. Engineered for resilience.</p>
        </footer> */}
      </div>
    </div>
  );
};

export default Layout;
