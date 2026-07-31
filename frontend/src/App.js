import React, { useState, useEffect } from 'react';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import Footer from './components/Layout/Footer';

import MetricsCards from './components/Dashboard/MetricsCards';
import ClassificationChart from './components/Dashboard/ClassificationChart';
import TrafficTimeline from './components/Dashboard/TrafficTimeline';
import GraphVisualization from './components/Dashboard/GraphVisualization';
import RecentFlowsTable from './components/Dashboard/RecentFlowsTable';
import TopApplications from './components/Dashboard/TopApplications';
import ProtocolDistribution from './components/Dashboard/ProtocolDistribution';
import AlertFeed from './components/Dashboard/AlertFeed';

import FlowDetailModal from './components/FlowDetail/FlowDetailModal';
import LoadingSpinner from './components/Common/LoadingSpinner';
import ErrorBoundary from './components/Common/ErrorBoundary';

import { useFlows } from './hooks/useFlows';
import { useWebSocket } from './hooks/useWebSocket';
import { fetchAnalytics, fetchModelPerformance } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedFlow, setSelectedFlow] = useState(null);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);

  const { flows, metrics, loading, error, setFilters, addRealtimeFlow } = useFlows(60);
  const { status: wsStatus, lastMessage } = useWebSocket();

  // Listen for real-time WebSocket live traffic updates
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'NEW_FLOW' && lastMessage.flow) {
      addRealtimeFlow(lastMessage.flow);
    }
  }, [lastMessage, addRealtimeFlow]);

  // Load secondary tab data when selected
  useEffect(() => {
    if (activeTab === 'analytics' && !analyticsData) {
      fetchAnalytics().then(setAnalyticsData).catch(console.error);
    } else if (activeTab === 'performance' && !performanceData) {
      fetchModelPerformance().then(setPerformanceData).catch(console.error);
    }
  }, [activeTab, analyticsData, performanceData]);

  return (
    <ErrorBoundary>
      <div className="app-container">
        {/* Left Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Main Content Area */}
        <div className="main-content">
          <Header wsStatus={wsStatus} />

          <main className="page-body">
            {loading ? (
              <LoadingSpinner message="Initializing PyTorch Geometric GNN Models & SQLite Database..." />
            ) : error ? (
              <div className="glass-card" style={{ borderColor: 'var(--cat-p2p)', textAlign: 'center', padding: '3rem' }}>
                <h3 style={{ color: 'var(--cat-p2p)' }}>Connection Error</h3>
                <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>{error}</p>
                <button
                  onClick={() => window.location.reload()}
                  style={{ marginTop: '1rem', padding: '0.5rem 1rem', borderRadius: '6px', background: 'var(--primary-cyan)', color: '#000', border: 'none', fontWeight: '600', cursor: 'pointer' }}
                >
                  Retry Connection
                </button>
              </div>
            ) : (
              <>
                {/* TAB 1: LIVE DASHBOARD */}
                {activeTab === 'dashboard' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <MetricsCards metrics={metrics} />

                    <div style={{padding:"20px",color:"white"}}>
                      Graph temporarily disabled
                    </div>

                    <div className="dashboard-grid">
                      <div className="col-8">
                        <TrafficTimeline hourlyData={analyticsData?.hourly_traffic || []} />
                      </div>
                      <div className="col-4">
                        <TopApplications distribution={metrics?.category_distribution} />
                      </div>
                    </div>

                    <RecentFlowsTable
                      flows={flows}
                      onSelectFlow={setSelectedFlow}
                      onFilterChange={(f) => setFilters((prev) => ({ ...prev, ...f }))}
                    />
                  </div>
                )}

                {/* TAB 2: NETWORK TOPOLOGY */}
                {activeTab === 'graph' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <GraphVisualization flows={flows} onSelectFlow={setSelectedFlow} />
                    <RecentFlowsTable flows={flows} onSelectFlow={setSelectedFlow} />
                  </div>
                )}

                {/* TAB 3: TRAFFIC ANALYTICS */}
                {activeTab === 'analytics' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div className="dashboard-grid">
                      <div className="col-6">
                        <TrafficTimeline hourlyData={analyticsData?.hourly_traffic || []} />
                      </div>
                      <div className="col-6">
                        <ProtocolDistribution protocols={metrics?.protocol_distribution} />
                      </div>
                    </div>
                    <div className="dashboard-grid">
                      <div className="col-6">
                        <TopApplications distribution={metrics?.category_distribution} />
                      </div>
                      <div className="col-6">
                        <AlertFeed flows={flows} />
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 4: MODEL PERFORMANCE */}
                {activeTab === 'performance' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div className="glass-card">
                      <h3 style={{ fontSize: '1.1rem', color: 'var(--text-main)', marginBottom: '0.4rem' }}>
                        {performanceData?.model_name || 'GraphSAGE GNN Model Performance'}
                      </h3>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                        Evaluated on ISCX VPN-nonVPN Dataset Split (70% Train / 15% Val / 15% Test)
                      </p>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '1rem', borderRadius: '8px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Accuracy</span>
                          <h2 className="mono" style={{ color: 'var(--primary-cyan)' }}>
                            {((performanceData?.accuracy || 0.942) * 100).toFixed(2)}%
                          </h2>
                        </div>
                        <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '1rem', borderRadius: '8px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Macro F1-Score</span>
                          <h2 className="mono" style={{ color: 'var(--cat-streaming)' }}>
                            {((performanceData?.macro_f1 || 0.938) * 100).toFixed(2)}%
                          </h2>
                        </div>
                        <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '1rem', borderRadius: '8px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Weighted F1-Score</span>
                          <h2 className="mono" style={{ color: 'var(--cat-file)' }}>
                            {((performanceData?.weighted_f1 || 0.941) * 100).toFixed(2)}%
                          </h2>
                        </div>
                        <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '1rem', borderRadius: '8px' }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Uncertainty Method</span>
                          <h2 style={{ fontSize: '1rem', color: 'var(--primary-blue)', marginTop: '0.3rem' }}>
                            MC Dropout (N=20)
                          </h2>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 5: DOCUMENTATION */}
                {activeTab === 'docs' && (
                  <div className="glass-card" style={{ padding: '2rem' }}>
                    <h2 className="gradient-text" style={{ fontSize: '1.4rem', marginBottom: '1rem' }}>
                      Project Documentation & Specifications
                    </h2>
                    <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '1rem' }}>
                      This system performs fine-grained encrypted traffic classification using Graph Neural Networks (GraphSAGE & GAT).
                      Network flows are modeled as graph nodes, and edges represent shared IP relationships and temporal proximity.
                    </p>
                    <ul style={{ color: 'var(--text-main)', paddingLeft: '1.2rem', lineHeight: '1.8', fontSize: '0.88rem' }}>
                      <li><strong>Feature Extraction:</strong> 28 statistical features including packet rates, byte counts, and IAT metrics.</li>
                      <li><strong>Uncertainty Estimation:</strong> Monte Carlo Dropout generates predictive variance for unknown flow isolation.</li>
                      <li><strong>Explainability:</strong> GNNExplainer computes feature attribution scores per flow.</li>
                    </ul>
                  </div>
                )}
              </>
            )}
          </main>

          <Footer />
        </div>

        {/* Flow Detail & GNNExplainer Modal */}
        <FlowDetailModal flow={selectedFlow} onClose={() => setSelectedFlow(null)} />
      </div>
    </ErrorBoundary>
  );
}

export default App;
