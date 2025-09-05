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
  }, [setNodes, setEdges, selectedNode]);

  // Generate code based on nodes and edges
  useEffect(() => {
    let generatedCode = 'import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\n';
    
    // Add imports based on node types
    const nodeTypes = nodes.map(node => node.data.label);
    
    if (nodeTypes.includes('Linear Regression')) {
      generatedCode += 'from sklearn.linear_model import LinearRegression\n';
    }
    if (nodeTypes.includes('Logistic Regression')) {
      generatedCode += 'from sklearn.linear_model import LogisticRegression\n';
    }
    if (nodeTypes.includes('Decision Tree')) {
      generatedCode += 'from sklearn.tree import DecisionTreeClassifier\n';
    }
    if (nodeTypes.includes('Random Forest')) {
      generatedCode += 'from sklearn.ensemble import RandomForestClassifier\n';
    }
    if (nodeTypes.includes('SVM')) {
      generatedCode += 'from sklearn.svm import SVC\n';
    }
    if (nodeTypes.includes('K-Means')) {
      generatedCode += 'from sklearn.cluster import KMeans\n';
    }
    
    generatedCode += '\n';
    
    // Add data loading
    if (nodeTypes.includes('Data Input')) {
      generatedCode += '# Load data\n';
      generatedCode += 'data = pd.read_csv("data.csv")\n';
      generatedCode += 'X = data.drop("target", axis=1)\n';
      generatedCode += 'y = data["target"]\n\n';
    }
    
    // Add preprocessing
    if (nodeTypes.includes('Preprocessing')) {
      generatedCode += '# Preprocessing\n';
      generatedCode += 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n';
    }
    
    // Add model training
    if (nodeTypes.includes('Linear Regression')) {
      generatedCode += '# Linear Regression\n';
      generatedCode += 'model = LinearRegression()\n';
      generatedCode += 'model.fit(X_train, y_train)\n\n';
    }
    
    if (nodeTypes.includes('Logistic Regression')) {
      generatedCode += '# Logistic Regression\n';
      generatedCode += 'model = LogisticRegression()\n';
      generatedCode += 'model.fit(X_train, y_train)\n\n';
    }
    
    if (nodeTypes.includes('Decision Tree')) {
      generatedCode += '# Decision Tree\n';
      generatedCode += 'model = DecisionTreeClassifier()\n';
      generatedCode += 'model.fit(X_train, y_train)\n\n';
    }
    
    if (nodeTypes.includes('Random Forest')) {
      generatedCode += '# Random Forest\n';
      generatedCode += 'model = RandomForestClassifier()\n';
      generatedCode += 'model.fit(X_train, y_train)\n\n';
    }
    
    if (nodeTypes.includes('SVM')) {
      generatedCode += '# SVM\n';
      generatedCode += 'model = SVC()\n';
      generatedCode += 'model.fit(X_train, y_train)\n\n';
    }
    
    if (nodeTypes.includes('K-Means')) {
      generatedCode += '# K-Means Clustering\n';
      generatedCode += 'model = KMeans(n_clusters=3)\n';
      generatedCode += 'model.fit(X)\n\n';
    }
    
    // Add evaluation
    if (nodeTypes.includes('Model Evaluation')) {
      generatedCode += '# Evaluation\n';
      generatedCode += 'score = model.score(X_test, y_test)\n';
      generatedCode += 'print(f"Model accuracy: {score}")\n';
    }
    
    setCode(generatedCode);
  }, [nodes, edges]);

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
          <Panel position="top-right">
            <button onClick={() => console.log('Execute pipeline')}>
              Execute Pipeline
            </button>
          </Panel>
        </ReactFlow>
      </div>
      <div className="right-panel">
        {selectedNode && (
          <div className="config-panel-container">
            <NodeConfigPanel node={selectedNode} />
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
