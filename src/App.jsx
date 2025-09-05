import React, { useState, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import './App.css';
import NodePanel from './components/NodePanel';
import NodeConfigPanel from './components/NodeConfigPanel';

// Initial nodes for demonstration
const initialNodes = [
  {
    id: '1',
    type: 'input',
    position: { x: 0, y: 0 },
    data: { label: 'Data Input' },
  },
  {
    id: '2',
    position: { x: 0, y: 100 },
    data: { label: 'Preprocessing' },
  },
];

const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [code, setCode] = useState('');
  const [nodeConfigs, setNodeConfigs] = useState({});

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const addNode = (nodeType) => {
    const newNode = {
      id: `node-${Date.now()}`,
      type: 'default',
      position: { x: Math.random() * 300, y: Math.random() * 300 },
      data: { label: nodeType },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const deleteNode = useCallback((nodeId) => {
    // Delete the node
    setNodes((nds) => nds.filter((node) => node.id !== nodeId));
    // Delete connected edges
    setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
    // Clear selection if the deleted node was selected
    if (selectedNode && selectedNode.id === nodeId) {
      setSelectedNode(null);
    }
    // Remove node config
    setNodeConfigs(prev => {
      const newConfigs = { ...prev };
      delete newConfigs[nodeId];
      return newConfigs;
    });
  }, [setNodes, setEdges, selectedNode]);

  const updateNodeConfig = (nodeId, config) => {
    setNodeConfigs(prev => ({
      ...prev,
      [nodeId]: config
    }));
  };

  // Generate code based on nodes, edges, and configurations
  useEffect(() => {
    // Build execution order based on connections
    const nodeMap = {};
    nodes.forEach(node => {
      nodeMap[node.id] = node;
    });

    // Find starting nodes (nodes with no incoming edges)
    const startingNodes = nodes.filter(node => 
      !edges.some(edge => edge.target === node.id)
    );

    // Build execution path
    const executionPath = [];
    const visited = new Set();
    
    const traverse = (nodeId) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      
      const node = nodeMap[nodeId];
      if (node) {
        executionPath.push(node);
      }
      
      // Find outgoing edges
      const outgoingEdges = edges.filter(edge => edge.source === nodeId);
      outgoingEdges.forEach(edge => {
        traverse(edge.target);
      });
    };
    
    // Start traversal from starting nodes
    startingNodes.forEach(node => {
      traverse(node.id);
    });

    // Generate code based on execution path
    let generatedCode = 'import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\n';
    
    // Add imports based on node types in execution order
    const addedImports = new Set();
    
    executionPath.forEach(node => {
      const config = nodeConfigs[node.id] || {};
      
      switch (node.data.label) {
        case 'Linear Regression':
          if (!addedImports.has('LinearRegression')) {
            generatedCode += 'from sklearn.linear_model import LinearRegression\n';
            addedImports.add('LinearRegression');
          }
          break;
        case 'Logistic Regression':
          if (!addedImports.has('LogisticRegression')) {
            generatedCode += 'from sklearn.linear_model import LogisticRegression\n';
            addedImports.add('LogisticRegression');
          }
          break;
        case 'Decision Tree':
          if (!addedImports.has('DecisionTreeClassifier')) {
            generatedCode += 'from sklearn.tree import DecisionTreeClassifier\n';
            addedImports.add('DecisionTreeClassifier');
          }
          break;
        case 'Random Forest':
          if (!addedImports.has('RandomForestClassifier')) {
            generatedCode += 'from sklearn.ensemble import RandomForestClassifier\n';
            addedImports.add('RandomForestClassifier');
          }
          break;
        case 'SVM':
          if (!addedImports.has('SVC')) {
            generatedCode += 'from sklearn.svm import SVC\n';
            addedImports.add('SVC');
          }
          break;
        case 'K-Means':
          if (!addedImports.has('KMeans')) {
            generatedCode += 'from sklearn.cluster import KMeans\n';
            addedImports.add('KMeans');
          }
          break;
        case 'Preprocessing':
          if (!addedImports.has('StandardScaler')) {
            generatedCode += 'from sklearn.preprocessing import StandardScaler, LabelEncoder\n';
            addedImports.add('StandardScaler');
          }
          break;
      }
    });
    
    generatedCode += '\n';
    
    // Add data loading (if Data Input node exists)
    if (executionPath.some(node => node.data.label === 'Data Input')) {
      generatedCode += '# Load data\n';
      generatedCode += 'data = pd.read_csv("data.csv")\n';
      generatedCode += 'X = data.drop("target", axis=1)\n';
      generatedCode += 'y = data["target"]\n\n';
    }
    
    // Add preprocessing (if Preprocessing node exists)
    if (executionPath.some(node => node.data.label === 'Preprocessing')) {
      generatedCode += '# Preprocessing\n';
      const preprocessNode = executionPath.find(node => node.data.label === 'Preprocessing');
      const config = nodeConfigs[preprocessNode?.id] || {};
      
      if (config.scaling === 'standard') {
        generatedCode += 'scaler = StandardScaler()\n';
        generatedCode += 'X_scaled = scaler.fit_transform(X)\n';
        generatedCode += 'X = X_scaled\n\n';
      }
      
      generatedCode += 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n';
    } else if (executionPath.some(node => 
      ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'].includes(node.data.label))) {
      // If we have a model but no preprocessing, still split the data
      generatedCode += 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n';
    }
    
    // Add model training based on execution order
    let modelVarName = null;
    executionPath.forEach(node => {
      const config = nodeConfigs[node.id] || {};
      
      switch (node.data.label) {
        case 'Linear Regression':
          generatedCode += '# Linear Regression\n';
          generatedCode += 'model = LinearRegression(';
          const lrParams = [];
          if (config.fitIntercept === false) lrParams.push('fit_intercept=False');
          if (config.normalize === true) lrParams.push('normalize=True');
          generatedCode += lrParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X_train, y_train)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
          
        case 'Logistic Regression':
          generatedCode += '# Logistic Regression\n';
          generatedCode += 'model = LogisticRegression(';
          const logrParams = [];
          if (config.penalty && config.penalty !== 'l2') logrParams.push(`penalty='${config.penalty}'`);
          if (config.solver && config.solver !== 'lbfgs') logrParams.push(`solver='${config.solver}'`);
          if (config.maxIter && config.maxIter !== 100) logrParams.push(`max_iter=${config.maxIter}`);
          generatedCode += logrParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X_train, y_train)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
          
        case 'Decision Tree':
          generatedCode += '# Decision Tree\n';
          generatedCode += 'model = DecisionTreeClassifier(';
          const dtParams = [];
          if (config.criterion && config.criterion !== 'gini') dtParams.push(`criterion='${config.criterion}'`);
          if (config.maxDepth) dtParams.push(`max_depth=${config.maxDepth}`);
          if (config.minSamplesSplit && config.minSamplesSplit !== 2) dtParams.push(`min_samples_split=${config.minSamplesSplit}`);
          generatedCode += dtParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X_train, y_train)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
          
        case 'Random Forest':
          generatedCode += '# Random Forest\n';
          generatedCode += 'model = RandomForestClassifier(';
          const rfParams = [];
          if (config.nEstimators && config.nEstimators !== 100) rfParams.push(`n_estimators=${config.nEstimators}`);
          if (config.criterion && config.criterion !== 'gini') rfParams.push(`criterion='${config.criterion}'`);
          if (config.maxDepth) rfParams.push(`max_depth=${config.maxDepth}`);
          generatedCode += rfParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X_train, y_train)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
          
        case 'SVM':
          generatedCode += '# SVM\n';
          generatedCode += 'model = SVC(';
          const svmParams = [];
          if (config.kernel && config.kernel !== 'rbf') svmParams.push(`kernel='${config.kernel}'`);
          if (config.C && config.C !== 1.0) svmParams.push(`C=${config.C}`);
          if (config.gamma && config.gamma !== 'scale') svmParams.push(`gamma='${config.gamma}'`);
          generatedCode += svmParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X_train, y_train)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
          
        case 'K-Means':
          generatedCode += '# K-Means Clustering\n';
          generatedCode += 'model = KMeans(';
          const kmParams = [];
          if (config.nClusters && config.nClusters !== 8) kmParams.push(`n_clusters=${config.nClusters}`);
          if (config.init && config.init !== 'k-means++') kmParams.push(`init='${config.init}'`);
          if (config.maxIter && config.maxIter !== 300) kmParams.push(`max_iter=${config.maxIter}`);
          generatedCode += kmParams.join(', ') + ')\n';
          generatedCode += 'model.fit(X)\n';
          modelVarName = 'model';
          generatedCode += '\n';
          break;
      }
    });
    
    // Add evaluation (if Model Evaluation node exists)
    if (executionPath.some(node => node.data.label === 'Model Evaluation') && modelVarName) {
      generatedCode += '# Evaluation\n';
      generatedCode += 'score = model.score(X_test, y_test)\n';
      generatedCode += 'print(f"Model accuracy: {score}")\n';
      
      const evalNode = executionPath.find(node => node.data.label === 'Model Evaluation');
      const config = nodeConfigs[evalNode?.id] || {};
      
      if (config.metrics && config.metrics.length > 0) {
        if (config.metrics.includes('precision')) {
          generatedCode += 'from sklearn.metrics import precision_score\n';
          generatedCode += 'precision = precision_score(y_test, model.predict(X_test))\n';
          generatedCode += 'print(f"Precision: {precision}")\n';
        }
        if (config.metrics.includes('recall')) {
          generatedCode += 'from sklearn.metrics import recall_score\n';
          generatedCode += 'recall = recall_score(y_test, model.predict(X_test))\n';
          generatedCode += 'print(f"Recall: {recall}")\n';
        }
        if (config.metrics.includes('f1')) {
          generatedCode += 'from sklearn.metrics import f1_score\n';
          generatedCode += 'f1 = f1_score(y_test, model.predict(X_test))\n';
          generatedCode += 'print(f"F1 Score: {f1}")\n';
        }
      }
    }
    
    setCode(generatedCode);
  }, [nodes, edges, nodeConfigs]);

  return (
    <div className="app">
      <div className="node-panel-container">
        <NodePanel onAddNode={addNode} />
      </div>
      <div className="main-content">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          fitView
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>
      <div className="right-panel">
        {selectedNode && (
          <div className="config-panel-container">
            <NodeConfigPanel 
              node={selectedNode} 
              config={nodeConfigs[selectedNode.id] || {}} 
              onUpdateConfig={(config) => updateNodeConfig(selectedNode.id, config)} 
            />
            <button onClick={() => deleteNode(selectedNode.id)} className="delete-button">
              Delete Node
            </button>
          </div>
        )}
        <div className="code-panel">
          <h3>Generated Code</h3>
          <pre className="code-display">{code}</pre>
        </div>
      </div>
    </div>
  );
}

export default App;
