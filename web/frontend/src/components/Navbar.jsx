import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="text-xl font-bold">UV & Cancer Assessment</Link>
          <div className="flex space-x-4">
            <Link to="/" className="hover:text-blue-200">Dashboard</Link>
            <Link to="/prediction" className="hover:text-blue-200">Predict & Forecast</Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
