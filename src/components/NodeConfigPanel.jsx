import React, { useState, useEffect } from 'react';

const NodeConfigPanel = ({ node, config, onUpdateConfig }) => {
  const [localConfig, setLocalConfig] = useState(config || {});

  useEffect(() => {
    setLocalConfig(config || {});
  }, [config]);

  const handleConfigChange = (key, value) => {
    const newConfig = { ...localConfig, [key]: value };
    setLocalConfig(newConfig);
    onUpdateConfig(newConfig);
  };

  const renderConfigForm = () => {
    switch (node.data.label) {
      case 'Data Input':
        return (
          <div className="config-form">
            <label>Source:</label>
            <select 
              value={localConfig.source || 'csv'} 
              onChange={(e) => handleConfigChange('source', e.target.value)}
            >
              <option value="csv">CSV File</option>
              <option value="json">JSON File</option>
              <option value="database">Database</option>
            </select>
            
            <label>File Path:</label>
            <input 
              type="text" 
              value={localConfig.filePath || ''} 
              onChange={(e) => handleConfigChange('filePath', e.target.value)} 
            />
            
            <label>Delimiter:</label>
            <input 
              type="text" 
              value={localConfig.delimiter || ','} 
              onChange={(e) => handleConfigChange('delimiter', e.target.value)} 
            />
            
            <label>Target Column:</label>
            <input 
              type="text" 
              value={localConfig.targetColumn || 'target'} 
              onChange={(e) => handleConfigChange('targetColumn', e.target.value)} 
              placeholder="Name of target column"
            />
          </div>
        );
      
      case 'Preprocessing':
        return (
          <div className="config-form">
            <label>Scaling:</label>
            <select 
              value={localConfig.scaling || 'none'} 
              onChange={(e) => handleConfigChange('scaling', e.target.value)}
            >
              <option value="none">None</option>
              <option value="standard">StandardScaler</option>
              <option value="minmax">MinMaxScaler</option>
              <option value="robust">RobustScaler</option>
              <option value="maxabs">MaxAbsScaler</option>
            </select>
            
            <label>Encoding:</label>
            <select 
              value={localConfig.encoding || 'none'} 
              onChange={(e) => handleConfigChange('encoding', e.target.value)}
            >
              <option value="none">None</option>
              <option value="onehot">One-Hot Encoding</option>
              <option value="label">Label Encoding</option>
              <option value="ordinal">Ordinal Encoding</option>
            </select>
            
            <label>Missing Values:</label>
            <select 
              value={localConfig.missingValues || 'drop'} 
              onChange={(e) => handleConfigChange('missingValues', e.target.value)}
            >
              <option value="drop">Drop Rows</option>
              <option value="mean">Fill with Mean</option>
              <option value="median">Fill with Median</option>
              <option value="knn">KNN Imputer</option>
            </select>
            
            <label>
              <input 
                type="checkbox" 
                checked={localConfig.featureSelection === true} 
                onChange={(e) => handleConfigChange('featureSelection', e.target.checked)} 
              />
              Feature Selection
            </label>
          </div>
        );
      
      case 'Scaling':
        return (
          <div className="config-form">
            <label>Scaling Method:</label>
            <select 
              value={localConfig.scalingMethod || 'standard'} 
              onChange={(e) => handleConfigChange('scalingMethod', e.target.value)}
            >
              <option value="standard">StandardScaler</option>
              <option value="minmax">MinMaxScaler</option>
              <option value="robust">RobustScaler</option>
              <option value="maxabs">MaxAbsScaler</option>
              <option value="normalizer">Normalizer</option>
            </select>
          </div>
        );
      
      case 'Encoding':
        return (
          <div className="config-form">
            <label>Encoding Method:</label>
            <select 
              value={localConfig.encodingMethod || 'onehot'} 
              onChange={(e) => handleConfigChange('encodingMethod', e.target.value)}
            >
              <option value="onehot">One-Hot Encoding</option>
              <option value="label">Label Encoding</option>
              <option value="ordinal">Ordinal Encoding</option>
            </select>
          </div>
        );
      
      case 'Missing Values':
        return (
          <div className="config-form">
            <label>Handling Method:</label>
            <select 
              value={localConfig.missingMethod || 'mean'} 
              onChange={(e) => handleConfigChange('missingMethod', e.target.value)}
            >
              <option value="mean">Fill with Mean</option>
              <option value="median">Fill with Median</option>
              <option value="mode">Fill with Mode</option>
              <option value="knn">KNN Imputer</option>
              <option value="drop">Drop Rows</option>
            </select>
          </div>
        );
      
      case 'Text Processing':
        return (
          <div className="config-form">
            <label>Text Processing Method:</label>
            <select 
              value={localConfig.textMethod || 'tfidf'} 
              onChange={(e) => handleConfigChange('textMethod', e.target.value)}
            >
              <option value="tfidf">TF-IDF Vectorizer</option>
              <option value="count">Count Vectorizer</option>
              <option value="hash">Hashing Vectorizer</option>
            </select>
          </div>
        );
      
      case 'Dimensionality Reduction':
        return (
          <div className="config-form">
            <label>Reduction Method:</label>
            <select 
              value={localConfig.reductionMethod || 'pca'} 
              onChange={(e) => handleConfigChange('reductionMethod', e.target.value)}
            >
              <option value="pca">PCA</option>
              <option value="svd">Truncated SVD</option>
              <option value="kpca">Kernel PCA</option>
            </select>
          </div>
        );
      
      case 'Linear Regression':
        return (
          <div className="config-form">
            <label>
              <input 
                type="checkbox" 
                checked={localConfig.fitIntercept !== false} 
                onChange={(e) => handleConfigChange('fitIntercept', e.target.checked)} 
              />
              Fit Intercept
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={localConfig.normalize === true} 
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
              value={localConfig.penalty || 'l2'} 
              onChange={(e) => handleConfigChange('penalty', e.target.value)}
            >
              <option value="l1">L1</option>
              <option value="l2">L2</option>
              <option value="elasticnet">ElasticNet</option>
              <option value="none">None</option>
            </select>
            
            <label>Solver:</label>
            <select 
              value={localConfig.solver || 'lbfgs'} 
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
              value={localConfig.maxIter || 100} 
              onChange={(e) => handleConfigChange('maxIter', parseInt(e.target.value))} 
            />
          </div>
        );
      
      case 'Decision Tree':
        return (
          <div className="config-form">
            <label>Criterion:</label>
            <select 
              value={localConfig.criterion || 'gini'} 
              onChange={(e) => handleConfigChange('criterion', e.target.value)}
            >
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
            </select>
            
            <label>Max Depth:</label>
            <input 
              type="number" 
              value={localConfig.maxDepth || ''} 
              onChange={(e) => handleConfigChange('maxDepth', e.target.value ? parseInt(e.target.value) : null)} 
              placeholder="None"
            />
            
            <label>Min Samples Split:</label>
            <input 
              type="number" 
              value={localConfig.minSamplesSplit || 2} 
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
              value={localConfig.nEstimators || 100} 
              onChange={(e) => handleConfigChange('nEstimators', parseInt(e.target.value))} 
            />
            
            <label>Criterion:</label>
            <select 
              value={localConfig.criterion || 'gini'} 
              onChange={(e) => handleConfigChange('criterion', e.target.value)}
            >
              <option value="gini">Gini</option>
              <option value="entropy">Entropy</option>
            </select>
            
            <label>Max Depth:</label>
            <input 
              type="number" 
              value={localConfig.maxDepth || ''} 
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
              value={localConfig.kernel || 'rbf'} 
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
              value={localConfig.C || 1.0} 
              onChange={(e) => handleConfigChange('C', parseFloat(e.target.value))} 
            />
            
            <label>Gamma:</label>
            <select 
              value={localConfig.gamma || 'scale'} 
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
              value={localConfig.nClusters || 8} 
              onChange={(e) => handleConfigChange('nClusters', parseInt(e.target.value))} 
            />
            
            <label>Initialization:</label>
            <select 
              value={localConfig.init || 'k-means++'} 
              onChange={(e) => handleConfigChange('init', e.target.value)}
            >
              <option value="k-means++">K-Means++</option>
              <option value="random">Random</option>
            </select>
            
            <label>Max Iterations:</label>
            <input 
              type="number" 
              value={localConfig.maxIter || 300} 
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
              value={localConfig.metrics || []} 
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
                checked={localConfig.crossValidation === true} 
                onChange={(e) => handleConfigChange('crossValidation', e.target.checked)} 
              />
              Cross Validation
            </label>
            
            {localConfig.crossValidation && (
              <>
                <label>CV Folds:</label>
                <input 
                  type="number" 
                  value={localConfig.cvFolds || 5} 
                  onChange={(e) => handleConfigChange('cvFolds', parseInt(e.target.value))} 
                />
              </>
            )}
          </div>
        );
      
      case 'Grid Search':
        return (
          <div className="config-form">
            <label>Parameter Grid:</label>
            <textarea 
              value={localConfig.paramGrid || ''} 
              onChange={(e) => handleConfigChange('paramGrid', e.target.value)} 
              placeholder="Enter parameter grid as JSON"
              rows="4"
            />
            
            <label>Cross Validation Folds:</label>
            <input 
              type="number" 
              value={localConfig.cvFolds || 5} 
              onChange={(e) => handleConfigChange('cvFolds', parseInt(e.target.value))} 
            />
            
            <label>Scoring Metric:</label>
            <select 
              value={localConfig.scoring || 'accuracy'} 
              onChange={(e) => handleConfigChange('scoring', e.target.value)}
            >
              <option value="accuracy">Accuracy</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="f1">F1 Score</option>
              <option value="roc_auc">ROC AUC</option>
            </select>
          </div>
        );
      
      case 'Random Search':
        return (
          <div className="config-form">
            <label>Parameter Distribution:</label>
            <textarea 
              value={localConfig.paramDist || ''} 
              onChange={(e) => handleConfigChange('paramDist', e.target.value)} 
              placeholder="Enter parameter distribution as JSON"
              rows="4"
            />
            
            <label>Number of Iterations:</label>
            <input 
              type="number" 
              value={localConfig.nIter || 10} 
              onChange={(e) => handleConfigChange('nIter', parseInt(e.target.value))} 
            />
            
            <label>Cross Validation Folds:</label>
            <input 
              type="number" 
              value={localConfig.cvFolds || 5} 
              onChange={(e) => handleConfigChange('cvFolds', parseInt(e.target.value))} 
            />
            
            <label>Scoring Metric:</label>
            <select 
              value={localConfig.scoring || 'accuracy'} 
              onChange={(e) => handleConfigChange('scoring', e.target.value)}
            >
              <option value="accuracy">Accuracy</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="f1">F1 Score</option>
              <option value="roc_auc">ROC AUC</option>
            </select>
          </div>
        );
      
      case 'PCA':
        return (
          <div className="config-form">
            <label>Number of Components:</label>
            <input 
              type="number" 
              value={localConfig.nComponents || 2} 
              onChange={(e) => handleConfigChange('nComponents', parseInt(e.target.value))} 
              placeholder="Number of components to keep"
            />
            
            <label>
              <input 
                type="checkbox" 
                checked={localConfig.whiten === true} 
                onChange={(e) => handleConfigChange('whiten', e.target.checked)} 
              />
              Whiten
            </label>
          </div>
        );
      
      case 'Feature Selection':
        return (
          <div className="config-form">
            <label>Method:</label>
            <select 
              value={localConfig.method || 'variance'} 
              onChange={(e) => handleConfigChange('method', e.target.value)}
            >
              <option value="variance">Variance Threshold</option>
              <option value="univariate">Univariate Selection</option>
              <option value="recursive">Recursive Feature Elimination</option>
            </select>
            
            <label>Threshold:</label>
            <input 
              type="number" 
              step="0.01"
              value={localConfig.threshold || 0.01} 
              onChange={(e) => handleConfigChange('threshold', parseFloat(e.target.value))} 
              placeholder="Feature selection threshold"
            />
          </div>
        );
      
      case 'Model Comparison':
        return (
          <div className="config-form">
            <label>Comparison Metrics:</label>
            <select 
              multiple
              value={localConfig.metrics || ['accuracy']} 
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
              <option value="mse">Mean Squared Error</option>
              <option value="mae">Mean Absolute Error</option>
            </select>
            
            <label>Visualization Type:</label>
            <select 
              value={localConfig.visualization || 'bar'} 
              onChange={(e) => handleConfigChange('visualization', e.target.value)}
            >
              <option value="bar">Bar Chart</option>
              <option value="radar">Radar Chart</option>
              <option value="table">Table</option>
            </select>
            
            <label>
              <input 
                type="checkbox" 
                checked={localConfig.statisticalTest === true} 
                onChange={(e) => handleConfigChange('statisticalTest', e.target.checked)} 
              />
              Include Statistical Significance Test
            </label>
          </div>
        );
      
      default:
        return (
          <div>
            <p>No configuration options available for this node type.</p>
            <p><small>Connect this node to others to see configuration options.</small></p>
          </div>
        );
    }
  };

  return (
    <div className="node-config-panel">
      <h3>Configure: {node.data.label}</h3>
      {renderConfigForm()}
    </div>
  );
};

export default NodeConfigPanel;
