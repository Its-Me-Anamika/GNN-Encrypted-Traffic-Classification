import React, { useEffect, useRef } from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data/peer/esm/vis-data';
import { getCategoryColor } from '../../utils/helpers';

export const GraphVisualization = ({ flows = [], onSelectFlow }) => {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || flows.length === 0) return;

    // Construct Vis.js Nodes from Traffic Flows
    const nodesArray = flows.slice(0, 35).map((flow, index) => {
      const category = flow.predicted_category || 'Unknown';
      const color = getCategoryColor(category);
      return {
        id: flow.id || index + 1,
        label: `${category}\n${flow.src_ip}`,
        shape: 'dot',
        size: 16 + (flow.total_packets % 12),
        color: {
          background: color,
          border: '#00f2fe',
          highlight: { background: '#00f2fe', border: '#ffffff' }
        },
        font: { color: '#f1f5f9', size: 10, face: 'Inter' },
        title: `Flow ID: ${flow.flow_id}<br/>Category: ${category}<br/>Src: ${flow.src_ip}:${flow.src_port}<br/>Dst: ${flow.dst_ip}:${flow.dst_port}<br/>Confidence: ${(flow.confidence * 100).toFixed(1)}%`,
        flowData: flow
      };
    });

    // Construct Edges based on shared IP addresses
    const edgesArray = [];
    const nodeMap = new Map();
    nodesArray.forEach((node) => nodeMap.set(node.id, node.flowData));

    const keys = Array.from(nodeMap.keys());
    for (let i = 0; i < keys.length; i++) {
      for (let j = i + 1; j < keys.length; j++) {
        const idA = keys[i];
        const idB = keys[j];
        const fA = nodeMap.get(idA);
        const fB = nodeMap.get(idB);

        if (fA.src_ip === fB.src_ip || fA.dst_ip === fB.dst_ip || fA.src_ip === fB.dst_ip) {
          edgesArray.push({
            from: idA,
            to: idB,
            color: { color: 'rgba(0, 242, 254, 0.3)', highlight: '#00f2fe' },
            width: 1.5
          });
        }
      }
    }

    const data = {
      nodes: new DataSet(nodesArray),
      edges: new DataSet(edgesArray)
    };

    const options = {
      nodes: {
        borderWidth: 1.5,
        shadow: true
      },
      edges: {
        smooth: { type: 'continuous' }
      },
      physics: {
        stabilization: false,
        barnesHut: {
          gravitationalConstant: -2000,
          springLength: 95,
          springConstant: 0.04
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100
      }
    };

    const network = new Network(containerRef.current, data, options);
    networkRef.current = network;

    network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const selectedId = params.nodes[0];
        const nodeObj = nodesArray.find((n) => n.id === selectedId);
        if (nodeObj && onSelectFlow) {
          onSelectFlow(nodeObj.flowData);
        }
      }
    });

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [flows, onSelectFlow]);

  return (
    <div className="glass-card" style={{ height: '420px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.8rem' }}>
        <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', fontWeight: '600' }}>
          Interactive Flow Graph Topology (PyG Node Adjacency)
        </h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          Nodes = 5-Tuple Flows | Edges = Shared IP Relationships
        </span>
      </div>
      <div ref={containerRef} style={{ flex: 1, borderRadius: '8px', background: 'rgba(7, 10, 18, 0.6)' }} />
    </div>
  );
};

export default GraphVisualization;
