import React, { useState } from 'react';
import { Search, Filter, Eye, ShieldCheck, HelpCircle } from 'lucide-react';
import { formatBytes, formatDuration, getCategoryBadgeClass } from '../../utils/helpers';

export const RecentFlowsTable = ({ flows = [], onSelectFlow, onFilterChange }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const categories = ['All', 'Streaming', 'Chat', 'VoIP', 'File Transfer', 'Browsing', 'P2P', 'Email', 'VPN', 'Unknown'];

  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };

  const handleCategorySelect = (cat) => {
    setSelectedCategory(cat);
    if (onFilterChange) {
      onFilterChange({ category: cat, search: searchTerm });
    }
  };

  const filteredFlows = flows.filter((flow) => {
    const matchesCategory = selectedCategory === 'All' || flow.predicted_category === selectedCategory;
    const matchesSearch = !searchTerm ||
      flow.src_ip.includes(searchTerm) ||
      flow.dst_ip.includes(searchTerm) ||
      flow.flow_id.includes(searchTerm) ||
      flow.protocol.includes(searchTerm);
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Header & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: '600' }}>
          Recent Classified Traffic Flows
        </h3>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', flexWrap: 'wrap' }}>
          {/* Search Box */}
          <div style={{ position: 'relative', minWidth: '200px' }}>
            <Search style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', width: '14px', height: '14px', color: 'var(--text-dim)' }} />
            <input
              type="text"
              placeholder="Search IP / Flow ID..."
              value={searchTerm}
              onChange={handleSearch}
              style={{
                width: '100%',
                padding: '0.4rem 0.6rem 0.4rem 2rem',
                borderRadius: '6px',
                background: 'rgba(7, 10, 18, 0.8)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-main)',
                fontSize: '0.85rem'
              }}
            />
          </div>

          {/* Category Filter Pills */}
          <div style={{ display: 'flex', gap: '0.3rem', overflowX: 'auto', paddingBottom: '0.2rem' }}>
            {categories.slice(0, 6).map((cat) => (
              <button
                key={cat}
                onClick={() => handleCategorySelect(cat)}
                style={{
                  padding: '0.3rem 0.6rem',
                  borderRadius: '6px',
                  border: 'none',
                  background: selectedCategory === cat ? 'var(--primary-cyan)' : 'rgba(255, 255, 255, 0.05)',
                  color: selectedCategory === cat ? '#000' : 'var(--text-muted)',
                  fontSize: '0.75rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap'
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Flows Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-dim)', height: '36px' }}>
              <th style={{ padding: '0.5rem' }}>Flow ID</th>
              <th style={{ padding: '0.5rem' }}>Source IP:Port</th>
              <th style={{ padding: '0.5rem' }}>Destination IP:Port</th>
              <th style={{ padding: '0.5rem' }}>Proto</th>
              <th style={{ padding: '0.5rem' }}>Volume</th>
              <th style={{ padding: '0.5rem' }}>Predicted Category</th>
              <th style={{ padding: '0.5rem' }}>Confidence</th>
              <th style={{ padding: '0.5rem', textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredFlows.length > 0 ? (
              filteredFlows.map((flow) => {
                const badgeClass = getCategoryBadgeClass(flow.predicted_category);
                return (
                  <tr
                    key={flow.id || flow.flow_id}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      transition: 'background 0.2s ease',
                      cursor: 'pointer'
                    }}
                    onClick={() => onSelectFlow(flow)}
                  >
                    <td className="mono" style={{ padding: '0.75rem 0.5rem', color: 'var(--primary-cyan)' }}>
                      {flow.flow_id}
                    </td>
                    <td className="mono" style={{ padding: '0.75rem 0.5rem', color: 'var(--text-main)' }}>
                      {flow.src_ip}:{flow.src_port}
                    </td>
                    <td className="mono" style={{ padding: '0.75rem 0.5rem', color: 'var(--text-main)' }}>
                      {flow.dst_ip}:{flow.dst_port}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-muted)' }}>
                      {flow.protocol}
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-muted)' }}>
                      {formatBytes(flow.total_bytes)} ({flow.total_packets} pkts)
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem' }}>
                      <span className={`badge ${badgeClass}`}>
                        {flow.is_vpn && <ShieldCheck style={{ width: '12px', height: '12px' }} />}
                        {flow.predicted_category}
                      </span>
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ flex: 1, height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', minWidth: '40px' }}>
                          <div
                            style={{
                              width: `${(flow.confidence * 100).toFixed(0)}%`,
                              height: '100%',
                              background: flow.is_unknown ? 'var(--cat-p2p)' : 'var(--primary-cyan)',
                              borderRadius: '2px'
                            }}
                          />
                        </div>
                        <span className="mono" style={{ fontSize: '0.75rem', color: flow.is_unknown ? 'var(--cat-p2p)' : 'var(--text-main)' }}>
                          {(flow.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectFlow(flow);
                        }}
                        style={{
                          background: 'rgba(0, 242, 254, 0.1)',
                          border: '1px solid rgba(0, 242, 254, 0.3)',
                          color: 'var(--primary-cyan)',
                          padding: '0.3rem 0.6rem',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.3rem'
                        }}
                      >
                        <Eye style={{ width: '12px', height: '12px' }} />
                        Explain
                      </button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan="8" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)' }}>
                  No matching traffic flows found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RecentFlowsTable;
