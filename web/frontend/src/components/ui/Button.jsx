import React from 'react';

const Button = ({ children, onClick, variant = 'primary', className = '', type = 'button', disabled = false }) => {
  const baseStyles = "px-4 py-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform active:scale-95";
  
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-500 text-white shadow-lg hover:shadow-blue-500/30",
    secondary: "bg-white/10 hover:bg-white/20 text-white border border-white/10",
    outline: "border-2 border-blue-500 text-blue-400 hover:bg-blue-500/10",
    danger: "bg-red-500/20 text-red-300 border border-red-500/50 hover:bg-red-500/30"
  };

  return (
    <button
      type={type}
      className={`${baseStyles} ${variants[variant]} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;
