import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 15000
});

export const fetchHealth = async () => {
  const res = await apiClient.get('/health');
  return res.data;
};

export const fetchMetrics = async () => {
  const res = await apiClient.get('/metrics');
  return res.data;
};

export const fetchFlows = async (params = {}) => {
  const res = await apiClient.get('/flows', { params });
  return res.data;
};

export const fetchFlowDetail = async (flowId) => {
  const res = await apiClient.get(`/flows/${flowId}`);
  return res.data;
};

export const classifyFlow = async (payload) => {
  const res = await apiClient.post('/classify', payload);
  return res.data;
};

export const fetchAnalytics = async () => {
  const res = await apiClient.get('/analytics');
  return res.data;
};

export const fetchModelPerformance = async () => {
  const res = await apiClient.get('/model/performance');
  return res.data;
};

export default apiClient;
