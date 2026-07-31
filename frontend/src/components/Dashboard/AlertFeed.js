import React from 'react';
import { AlertCircle, ShieldAlert, CheckCircle2 } from 'lucide-react';

export const AlertFeed = ({ flows = [] }) => {
  const unknownFlows = flows.filter((f) => f.is_unknown || f.confidence < 0.65).slice(0, 4);

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <ShieldAlert style={{ width: '16px', height: '16px', color: 'var(--cat-p2p)' }} />
        Security & Anomaly Feed
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {unknownFlows.length > 0 ? (
          unknownFlows.map((flow, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.6rem 0.8rem',
                borderRadius: '6px',
                background: 'rgba(239, 68, 68, 0.08)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
                fontSize: '0.78rem'
              }}
            >
              <AlertCircle style={{ width: '16px', height: '16px', color: 'var(--cat-p2p)', flexShrink: 0 }} />
              <div>
                <p style={{ color: 'var(--text-main)', fontWeight: '600' }}>Low Confidence / Out-of-Distribution Flow</p>
                <p className="mono" style={{ color: 'var(--text-dim)', fontSize: '0.72rem' }}>
                  {flow.src_ip} &rarr; {flow.dst_ip} (Conf: {(flow.confidence * 100).toFixed(0)}%)
                </p>
              </div>
            </div>
          ))
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.8rem', color: 'var(--cat-file)', fontSize: '0.8rem' }}>
            <CheckCircle2 style={{ width: '16px', height: '16px' }} />
            <span>All classified flows meet high confidence threshold (&gt;65%)</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default AlertFeed;
