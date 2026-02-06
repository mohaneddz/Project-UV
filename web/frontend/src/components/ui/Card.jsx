import React from 'react';

const Card = ({ children, className = '', title }) => {
  return (
    <div className={`glass-panel rounded-2xl p-6 relative overflow-hidden group ${className}`}>
      {/* Subtle gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
      
      {title && (
        <h3 className="text-sm font-semibold mb-6 text-gray-300 uppercase tracking-wider flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shadow-[0_0_8px_rgba(129,140,248,0.8)]"></span>
          {title}
        </h3>
      )}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

export default Card;
