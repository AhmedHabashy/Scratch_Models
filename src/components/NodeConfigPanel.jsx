import React, { useState, useEffect } from 'react';

const NodeConfigPanel = ({ node, onDelete }) => {
  const [config, setConfig] = useState({});

  useEffect(() => {
    // Initialize config based on node type
    switch (node.data.label) {
      case 'Data Input':
        setConfig({
          source: 'csv',
          filePath: '',
          delimiter: ',',
        });
        break;
      case 'Preprocessing':
        setConfig({
          scaling: 'none',
          encoding: 'none',
          missingValues: 'drop',
        });
        break;
      case 'Linear Regression':
        setConfig({
          fitIntercept: true,
          normalize: false,
        });
        break;
      case 'Logistic Regression':
        setConfig({
          penalty: 'l2',
          solver: 'lbfgs',
          maxIter: 100,
        });
        break;
      case 'Decision Tree':
        setConfig({
          criterion: 'gini',
          maxDepth: null,
          minSamplesSplit: 2,
        });
        break;
      case 'Random Forest':
        setConfig({
          nEstimators: 100,
          criterion: 'gini',
          maxDepth: null,
        });
        break;
      case 'SVM':
        setConfig({
          kernel: 'rbf',
          C: 1.0,
          gamma: 'scale',
        });
        break;
      case 'K-Means':
        setConfig({
          nClusters: 8,
          init: 'k-means++',
          maxIter: 300,
        });
        break;
      case 'Model Evaluation':
        setConfig({
          metrics: ['accuracy'],
          crossValidation: false,
          cvFolds: 5,
        });
        break;
      default:
        setConfig({});
    }
  }, [node]);

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const renderConfigForm = () => {
    switch (node.data.label) {
      case 'Data Input':
        return (
          <div className="config-form">
            <label>Source:</label>
            <select 
              value={config.source || ''} 
              onChange={(e) => handleConfigChange('source', e.target.value)}
            >
              <option value="csv">CSV File</option>
              <option value="json">JSON File</option>
              <option value="database">Database</option>
            </select>
            
            <label>File Path:</label>
            <input 
              type="text" 
              value={config.filePath || ''} 
              onChange={(e) => handleConfigChange('filePath', e.target.value)} 
            />
            
            <label>Delimiter:</label>
            <input 
              type="text" 
              value={config.delimiter || ','} 
              onChange={(e) => handleConfigChange('delimiter', e.target.value)} 
            />
          </div>
        );
      
      case 'Preprocessing':
        return (
          <div className="config-form">
            <label>Scaling:</label>
            <select 
              value={config.scaling || 'none'} 
              onChange={(e) => handleConfigChange('scaling', e.target.value)}
            >
              <option value="none">None</option>
              <option value="standard">StandardScaler</option>
              <option value="minmax">MinMaxScaler</option>
              <option value="robust">RobustScaler</option>
            </select>
            
            <label>Encoding:</label>
            <select 
              value={config.encoding || 'none'} 
              onChange={(e) => handleConfigChange('encoding', e.target.value)}
            >
              <option value="none">None</option>
              <option value="onehot">One-Hot Encoding</option>
              <option value="label">Label Encoding</option>
            </select>
            
            <label>Missing Values:</label>
            <select 
              value={config.missingValues || 'drop'} 
              onChange={(e) => handleConfigChange('missingValues', e.target.value)}
            >
              <option value="drop">Drop Rows</option>
              <option value="mean">Fill with Mean</option>
              <option value="median">Fill with Median</option>
            </select>
          </div>
        );
      
      case 'Linear Regression':
        return (
          <div className="config-form">
            <label>
              <input 
                type="checkbox" 
                checked={config.fitIntercept || true} 
                onChange={(e) => handleConfigChange('fitIntercept', e.target.checked)} 
              />
              Fit Intercept
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={config.normalize || false} 
                onChange={(e) => handleConfigChange('normalize', e.target.checked)} 
              />
              Normalize
            </label>
          </div>
        );
      
      case 'Logistic Regression':
        return (
          <div className="config-form">
            <label>Penalty:</label>
            <select 
              value={config.penalty || 'l2'} 
              onChange={(e) => handleConfigChange('penalty', e.target.value)}
            >
              <option value="l1">L1</option>
              <option value="l2">L2</option>
              <option value="elasticnet">ElasticNet</option>
              <option value="none">None</option>
            </select>
            
            <label>Solver:</label>
            <select 
              value={config.solver || 'lbfgs'} 
              onChange={(e) => handleConfigChange('solver', e.target.value)}
            >
              <option value="lbfgs">lbfgs</option>
              <option value="liblinear">liblinear</option>
              <option value="sag">sag</option>
              <option value="saga">saga</option>
            </select>
            
            <label>Max Iterations:</label>
            <input 
              type="number" 
              value={config.maxIter || 100} 
              onChange={(e) => handleConfigChange('maxIter', parseInt(e.target.value))} 
            />
          </div>
        );
      
      case 'Decision Tree':
        return (
          <div className="config-form">
            <label>Criterion:</label>
            <select 
              value={config.criterion || 'gini'} 
              onChange={(e) => handleConfigChange('criterion', e.target.value)}
            >
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
            </select>
            
            <label>Max Depth:</label>
            <input 
              type="number" 
              value={config.maxDepth || ''} 
              onChange={(e) => handleConfigChange('maxDepth', e.target.value ? parseInt(e.target.value) : null)} 
              placeholder="None"
            />
            
            <label>Min Samples Split:</label>
            <input 
              type="number" 
              value={config.minSamplesSplit || 2} 
              onChange={(e) => handleConfigChange('minSamplesSplit', parseInt(e.target.value))} 
            />
          </div>
        );
      
      case 'Random Forest':
        return (
          <div className="config-form">
            <label>Number of Estimators:</label>
            <input 
              type="number" 
              value={config.nEstimators || 100} 
              onChange={(e) => handleConfigChange('nEstimators', parseInt(e.target.value))} 
            />
            
            <label>Criterion:</label>
            <select 
              value={config.criterion || 'gini'} 
              onChange={(e) => handleConfigChange('criterion', e.target.value)}
            >
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
            </select>
            
            <label>Max Depth:</label>
            <input 
              type="number" 
              value={config.maxDepth || ''} 
              onChange={(e) => handleConfigChange('maxDepth', e.target.value ? parseInt(e.target.value) : null)} 
              placeholder="None"
            />
          </div>
        );
      
      case 'SVM':
        return (
          <div className="config-form">
            <label>Kernel:</label>
            <select 
              value={config.kernel || 'rbf'} 
              onChange={(e) => handleConfigChange('kernel', e.target.value)}
            >
              <option value="linear">Linear</option>
              <option value="poly">Polynomial</option>
              <option value="rbf">RBF</option>
              <option value="sigmoid">Sigmoid</option>
            </select>
            
            <label>Regularization (C):</label>
            <input 
              type="number" 
              step="0.1"
              value={config.C || 1.0} 
              onChange={(e) => handleConfigChange('C', parseFloat(e.target.value))} 
            />
            
            <label>Gamma:</label>
            <select 
              value={config.gamma || 'scale'} 
              onChange={(e) => handleConfigChange('gamma', e.target.value)}
            >
              <option value="scale">Scale</option>
              <option value="auto">Auto</option>
            </select>
          </div>
        );
      
      case 'K-Means':
        return (
          <div className="config-form">
            <label>Number of Clusters:</label>
            <input 
              type="number" 
              value={config.nClusters || 8} 
              onChange={(e) => handleConfigChange('nClusters', parseInt(e.target.value))} 
            />
            
            <label>Initialization:</label>
            <select 
              value={config.init || 'k-means++'} 
              onChange={(e) => handleConfigChange('init', e.target.value)}
            >
              <option value="k-means++">K-Means++</option>
              <option value="random">Random</option>
            </select>
            
            <label>Max Iterations:</label>
            <input 
              type="number" 
              value={config.maxIter || 300} 
              onChange={(e) => handleConfigChange('maxIter', parseInt(e.target.value))} 
            />
          </div>
        );
      
      case 'Model Evaluation':
        return (
          <div className="config-form">
            <label>Metrics:</label>
            <select 
              multiple
              value={config.metrics || ['accuracy']} 
              onChange={(e) => {
                const selected = Array.from(e.target.selectedOptions, option => option.value);
                handleConfigChange('metrics', selected);
              }}
            >
              <option value="accuracy">Accuracy</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="f1">F1 Score</option>
              <option value="roc_auc">ROC AUC</option>
            </select>
            
            <label>
              <input 
                type="checkbox" 
                checked={config.crossValidation || false} 
                onChange={(e) => handleConfigChange('crossValidation', e.target.checked)} 
              />
              Cross Validation
            </label>
            
            {config.crossValidation && (
              <>
                <label>CV Folds:</label>
                <input 
                  type="number" 
                  value={config.cvFolds || 5} 
                  onChange={(e) => handleConfigChange('cvFolds', parseInt(e.target.value))} 
                />
              </>
            )}
          </div>
        );
      
      default:
        return <p>No configuration options available for this node type.</p>;
    }
  };

  return (
    <div className="node-config-panel">
      <h3>Configure: {node.data.label}</h3>
      {renderConfigForm()}
      <button onClick={() => console.log('Save configuration', config)}>
        Save Configuration
      </button>
    </div>
  );
};

export default NodeConfigPanel;
