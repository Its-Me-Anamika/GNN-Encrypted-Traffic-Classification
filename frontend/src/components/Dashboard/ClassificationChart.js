import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { getCategoryColor } from '../../utils/helpers';

ChartJS.register(ArcElement, Tooltip, Legend);

export const ClassificationChart = ({ distribution = {} }) => {
  const labels = Object.keys(distribution);
  const dataValues = Object.values(distribution);
  const backgroundColors = labels.map((label) => getCategoryColor(label));

  const chartData = {
    labels,
    datasets: [
      {
        data: dataValues,
        backgroundColor: backgroundColors,
        borderColor: '#0b0f19',
        borderWidth: 2,
        hoverOffset: 6
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          color: '#94a3b8',
          font: { size: 11, family: 'Inter' },
          boxWidth: 12,
          padding: 10
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => ` ${context.label}: ${context.raw} flows`
        }
      }
    },
    cutout: '70%'
  };

  return (
    <div className="glass-card" style={{ height: '340px', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ fontSize: '0.95rem', color: 'var(--text-main)', marginBottom: '1rem', fontWeight: '600' }}>
        Fine-Grained Classification Breakdown
      </h3>
      <div style={{ flex: 1, position: 'relative' }}>
        {labels.length > 0 ? (
          <Doughnut data={chartData} options={options} />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-dim)', fontSize: '0.85rem' }}>
            No classification data available
          </div>
        )}
      </div>
    </div>
  );
};

export default ClassificationChart;
