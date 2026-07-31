import React from 'react';
import { Cpu, Zap } from 'lucide-react';

export const ExplainabilityPanel = ({ explanation }) => {
  if (!explanation || !explanation.top_features) {
    return (
      <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.85rem' }}>
        No GNNExplainer attribution computed for this flow.
      </div>
    );
  }

  const topFeatures = explanation.top_features || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div
        style={{
          background: 'rgba(0, 242, 254, 0.05)',
          border: '1px solid rgba(0, 242, 254, 0.2)',
          borderRadius: '8px',
          padding: '0.8rem 1rem',
          fontSize: '0.82rem',
          color: 'var(--text-main)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem'
        }}
      >
        <Zap style={{ width: '18px', height: '18px', color: 'var(--primary-cyan)', flexShrink: 0 }} />
        <span>{explanation.explanation_summary}</span>
      </div>

      <h4 style={{ fontSize: '0.88rem', color: 'var(--text-main)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
        <Cpu style={{ width: '16px', height: '16px', color: 'var(--cat-streaming)' }} />
        GNNExplainer Feature Importance Attributions
      </h4>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {topFeatures.map((feat, idx) => (
          <div key={idx} style={{ background: 'rgba(7, 10, 18, 0.6)', padding: '0.6rem 0.8rem', borderRadius: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.3rem' }}>
              <span className="mono" style={{ color: 'var(--primary-cyan)', fontWeight: '500' }}>
                {feat.feature_name}
              </span>
              <span className="mono" style={{ color: 'var(--text-muted)' }}>
                Score: {feat.importance_score}% (Val: {feat.raw_value})
              </span>
            </div>
            <div style={{ height: '5px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, feat.importance_score * 2.5)}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, var(--primary-cyan), var(--primary-blue))',
                  borderRadius: '3px'
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplainabilityPanel;
