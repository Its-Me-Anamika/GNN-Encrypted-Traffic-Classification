import React from 'react';
import { AlertTriangle } from 'lucide-react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an exception:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="glass-card" style={{ margin: '2rem', padding: '2rem', textAlign: 'center', borderColor: 'var(--cat-p2p)' }}>
          <AlertTriangle style={{ width: '48px', height: '48px', color: 'var(--cat-p2p)', margin: '0 auto 1rem' }} />
          <h2 style={{ color: 'var(--text-main)', marginBottom: '0.5rem' }}>Dashboard Component Error</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem' }}>
            {this.state.error?.message || 'An unexpected error occurred while rendering the visualization.'}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              background: 'linear-gradient(135deg, var(--primary-cyan), var(--primary-blue))',
              border: 'none',
              padding: '0.6rem 1.2rem',
              borderRadius: '6px',
              color: '#000',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Reload Dashboard
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
