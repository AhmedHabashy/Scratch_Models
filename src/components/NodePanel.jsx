import React from 'react';

const nodeTypes = [
  'Data Input',
  'Preprocessing',
  'Model Evaluation',
  'Grid Search',
  'Random Search',
  'Model Comparison',
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
      
      <h3>Advanced Preprocessing</h3>
      <button className="node-button" onClick={() => onAddNode('PCA')}>
        PCA
      </button>
      <button className="node-button" onClick={() => onAddNode('Feature Selection')}>
        Feature Selection
      </button>
      <button className="node-button" onClick={() => onAddNode('Scaling')}>
        Scaling
      </button>
      <button className="node-button" onClick={() => onAddNode('Encoding')}>
        Encoding
      </button>
      <button className="node-button" onClick={() => onAddNode('Missing Values')}>
        Missing Values
      </button>
      <button className="node-button" onClick={() => onAddNode('Text Processing')}>
        Text Processing
      </button>
      <button className="node-button" onClick={() => onAddNode('Dimensionality Reduction')}>
        Dimensionality Reduction
      </button>
    </div>
  );
};

export default NodePanel;
