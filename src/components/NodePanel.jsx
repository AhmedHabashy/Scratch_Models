import React from 'react';

const nodeTypes = [
  'Data Input',
  'Preprocessing',
  'Model Training',
  'Model Evaluation',
  'Visualization',
  'Export Results',
];

const NodePanel = ({ onAddNode }) => {
  return (
    <div className="node-panel">
      <h3>Node Types</h3>
      {nodeTypes.map((nodeType) => (
        <button
          key={nodeType}
          className="node-button"
          onClick={() => onAddNode(nodeType)}
        >
          {nodeType}
        </button>
      ))}
      
      <h3>Sklearn Models</h3>
      <button className="node-button" onClick={() => onAddNode('Linear Regression')}>
        Linear Regression
      </button>
      <button className="node-button" onClick={() => onAddNode('Logistic Regression')}>
        Logistic Regression
      </button>
      <button className="node-button" onClick={() => onAddNode('Decision Tree')}>
        Decision Tree
      </button>
      <button className="node-button" onClick={() => onAddNode('Random Forest')}>
        Random Forest
      </button>
      <button className="node-button" onClick={() => onAddNode('SVM')}>
        SVM
      </button>
      <button className="node-button" onClick={() => onAddNode('K-Means')}>
        K-Means Clustering
      </button>
    </div>
  );
};

export default NodePanel;
