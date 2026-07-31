import { useState, useEffect, useCallback } from 'react';
import { fetchFlows, fetchMetrics } from '../services/api';

export const useFlows = (initialLimit = 50) => {
  const [flows, setFlows] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({ category: 'All', search: '', page: 1, limit: initialLimit });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [flowsData, metricsData] = await Promise.all([
        fetchFlows(filters),
        fetchMetrics()
      ]);
      setFlows(flowsData.flows || []);
      setMetrics(metricsData);
      setError(null);
    } catch (err) {
      console.error('Error fetching flows/metrics:', err);
      setError('Failed to connect to backend service.');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const addRealtimeFlow = useCallback((newFlow) => {
    setFlows((prev) => [newFlow, ...prev.slice(0, filters.limit - 1)]);
    setMetrics((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        total_flows: prev.total_flows + 1,
        classified_today: prev.classified_today + 1
      };
    });
  }, [filters.limit]);

  return { flows, metrics, loading, error, filters, setFilters, refresh: loadData, addRealtimeFlow };
};
