import React from 'react';
import { X, ShieldCheck, Activity, Cpu, Network } from 'lucide-react';
import ExplainabilityPanel from './ExplainabilityPanel';
import { formatBytes, formatDuration, getCategoryBadgeClass } from '../../utils/helpers';

export const FlowDetailModal = ({ flow, onClose }) => {
  if (!flow) return null;

  const badgeClass = getCategoryBadgeClass(flow.predicted_category);

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(8px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem'
      }}
      onClick={onClose}
    >
      <div
        className="glass-card"
        style={{
          width: '100%',
          maxWidth: '850px',
          maxHeight: '90vh',
          overflowY: 'auto',
          padding: '1.8rem',
          position: 'relative',
          border: '1px solid var(--border-highlight)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '1.2rem',
            right: '1.2rem',
            background: 'rgba(255, 255, 255, 0.05)',
            border: 'none',
            color: 'var(--text-muted)',
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <X style={{ width: '18px', height: '18px' }} />
        </button>

        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ padding: '0.8rem', borderRadius: '12px', background: 'rgba(0, 242, 254, 0.1)', color: 'var(--primary-cyan)' }}>
            <Activity style={{ width: '24px', height: '24px' }} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
              <h2 className="mono" style={{ fontSize: '1.3rem', color: 'var(--text-main)' }}>{flow.flow_id}</h2>
              <span className={`badge ${badgeClass}`}>
                {flow.is_vpn && <ShieldCheck style={{ width: '12px', height: '12px' }} />}
                {flow.predicted_category}
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
              5-Tuple Flow Details & GNN Graph Attribution
            </p>
          </div>
        </div>

        {/* 5-Tuple Network Metadata */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '0.8rem', borderRadius: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Source Address</span>
            <p className="mono" style={{ fontSize: '0.9rem', color: 'var(--primary-cyan)', marginTop: '0.2rem' }}>
              {flow.src_ip}:{flow.src_port}
            </p>
          </div>
          <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '0.8rem', borderRadius: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Destination Address</span>
            <p className="mono" style={{ fontSize: '0.9rem', color: 'var(--primary-blue)', marginTop: '0.2rem' }}>
              {flow.dst_ip}:{flow.dst_port}
            </p>
          </div>
          <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '0.8rem', borderRadius: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Protocol & Volume</span>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-main)', marginTop: '0.2rem' }}>
              {flow.protocol} &bull; {formatBytes(flow.total_bytes)}
            </p>
          </div>
          <div style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '0.8rem', borderRadius: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>MC Dropout Confidence</span>
            <p className="mono" style={{ fontSize: '0.9rem', color: 'var(--cat-file)', marginTop: '0.2rem' }}>
              {(flow.confidence * 100).toFixed(1)}% (Uncertainty: {flow.uncertainty_score})
            </p>
          </div>
        </div>

        {/* GNNExplainer Section */}
        <ExplainabilityPanel explanation={flow.explanation} />
      </div>
    </div>
  );
};

export default FlowDetailModal;
