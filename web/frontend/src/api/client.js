import axios from 'axios'

const API_BASE_URL = 'http://localhost:5000/api'

export const fetchCountries = async () => {
  const response = await axios.get(`${API_BASE_URL}/countries`)
  return response.data
}

export const fetchForecast = async (country = 'Algeria') => {
  const response = await axios.get(`${API_BASE_URL}/forecast`, {
    params: { country }
  })
  return response.data
}

export const submitPrediction = async (data) => {
  const response = await axios.post(`${API_BASE_URL}/predict/uv`, data)
  return response.data
}
