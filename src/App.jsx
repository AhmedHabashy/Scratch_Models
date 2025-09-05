import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
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
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

import './App.css';
import NodePanel from './components/NodePanel';
import NodeConfigPanel from './components/NodeConfigPanel';
import CustomNode from './components/CustomNode';

// Initial nodes for demonstration
const initialNodes = [
  {
    id: '1',
    type: 'custom',
    position: { x: 0, y: 0 },
    data: { label: 'Data Input' },
  },
  {
    id: '2',
    type: 'custom',
    position: { x: 0, y: 100 },
    data: { label: 'Preprocessing' },
  },
];

const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

function App() {
  const [rightPanelWidth, setRightPanelWidth] = useState(300);

  const handleResizeMouseDown = (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = rightPanelWidth;

    const handleMouseMove = (e) => {
      const newWidth = startWidth + (startX - e.clientX);
      setRightPanelWidth(Math.max(200, newWidth));
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [code, setCode] = useState('');
  const [nodeConfigs, setNodeConfigs] = useState({});
  const [results, setResults] = useState(null);
  const [plot, setPlot] = useState(null);
  const [projectName, setProjectName] = useState('MyProject');
  const fileInputRef = useRef(null);

  // Define node types
  const nodeTypes = useMemo(() => ({
    custom: CustomNode,
  }), []);

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
      type: 'custom',
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
    // Build execution order based on connections using topological sort
    const nodeMap = {};
    nodes.forEach(node => {
      nodeMap[node.id] = node;
    });

    // Build adjacency list and in-degree count
    const adjacencyList = {};
    const inDegree = {};
    
    // Initialize in-degree count for all nodes
    nodes.forEach(node => {
      inDegree[node.id] = 0;
      adjacencyList[node.id] = [];
    });
    
    // Build adjacency list and calculate in-degrees based on edges
    edges.forEach(edge => {
      if (!adjacencyList[edge.source]) {
        adjacencyList[edge.source] = [];
      }
      adjacencyList[edge.source].push(edge.target);
      inDegree[edge.target] = (inDegree[edge.target] || 0) + 1;
    });

    // Topological sort using Kahn's algorithm
    const executionPath = [];
    const queue = [];
    
    // Find all nodes with no incoming edges
    nodes.forEach(node => {
      if (inDegree[node.id] === 0) {
        queue.push(node);
      }
    });
    
    while (queue.length > 0) {
      const node = queue.shift();
      executionPath.push(node);
      
      // Update in-degrees of neighbors
      if (adjacencyList[node.id]) {
        adjacencyList[node.id].forEach(neighborId => {
          inDegree[neighborId]--;
          if (inDegree[neighborId] === 0) {
            const neighborNode = nodeMap[neighborId];
            if (neighborNode) {
              queue.push(neighborNode);
            }
          }
        });
      }
    }
    

    // Generate code based on execution path with proper data flow
    let generatedCode = 'import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\n';
    
    // Add imports based on node types in execution order
    const addedImports = new Set();
    
    // Create variable naming system for multiple nodes of the same type
    const nodeVariableCounts = {};
    const nodeVariables = {};
    
    // Count occurrences of each node type
    executionPath.forEach(node => {
      const label = node.data.label;
      nodeVariableCounts[label] = (nodeVariableCounts[label] || 0) + 1;
    });
    
    // Create variable names for each node
    const nodeTypeCounts = {};
    executionPath.forEach(node => {
      const label = node.data.label;
      const cleanLabel = label.replace(/\s+/g, '_').toLowerCase();
      nodeTypeCounts[label] = (nodeTypeCounts[label] || 0) + 1;
      const varName = nodeVariableCounts[label] > 1 ? 
        `${cleanLabel}_${nodeTypeCounts[label]}` : cleanLabel;
      nodeVariables[node.id] = varName;
    });
    
    executionPath.forEach(node => {
      const config = nodeConfigs[node.id] || {};
      const varName = nodeVariables[node.id];
      
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
        case 'Grid Search':
          if (!addedImports.has('GridSearchCV')) {
            generatedCode += 'from sklearn.model_selection import GridSearchCV\n';
            addedImports.add('GridSearchCV');
          }
          break;
        case 'Random Search':
          if (!addedImports.has('RandomizedSearchCV')) {
            generatedCode += 'from sklearn.model_selection import RandomizedSearchCV\n';
            addedImports.add('RandomizedSearchCV');
          }
          break;
        case 'PCA':
          if (!addedImports.has('PCA')) {
            generatedCode += 'from sklearn.decomposition import PCA\n';
            addedImports.add('PCA');
          }
          break;
        case 'Feature Selection':
          if (!addedImports.has('VarianceThreshold')) {
            generatedCode += 'from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE\n';
            addedImports.add('VarianceThreshold');
          }
          break;
        case 'Scaling':
          // Add imports based on scaling method
          if (!addedImports.has('StandardScaler')) {
            generatedCode += 'from sklearn.preprocessing import StandardScaler\n';
            addedImports.add('StandardScaler');
          }
          if (config.scalingMethod === 'minmax' && !addedImports.has('MinMaxScaler')) {
            generatedCode += 'from sklearn.preprocessing import MinMaxScaler\n';
            addedImports.add('MinMaxScaler');
          }
          if (config.scalingMethod === 'robust' && !addedImports.has('RobustScaler')) {
            generatedCode += 'from sklearn.preprocessing import RobustScaler\n';
            addedImports.add('RobustScaler');
          }
          if (config.scalingMethod === 'maxabs' && !addedImports.has('MaxAbsScaler')) {
            generatedCode += 'from sklearn.preprocessing import MaxAbsScaler\n';
            addedImports.add('MaxAbsScaler');
          }
          if (config.scalingMethod === 'normalizer' && !addedImports.has('Normalizer')) {
            generatedCode += 'from sklearn.preprocessing import Normalizer\n';
            addedImports.add('Normalizer');
          }
          break;
        case 'Encoding':
          // Add imports based on encoding method
          if (!addedImports.has('OneHotEncoder')) {
            generatedCode += 'from sklearn.preprocessing import OneHotEncoder\n';
            addedImports.add('OneHotEncoder');
          }
          if ((config.encodingMethod === 'label' || config.encodingMethod === 'ordinal') && !addedImports.has('LabelEncoder')) {
            generatedCode += 'from sklearn.preprocessing import LabelEncoder\n';
            addedImports.add('LabelEncoder');
          }
          if (config.encodingMethod === 'ordinal' && !addedImports.has('OrdinalEncoder')) {
            generatedCode += 'from sklearn.preprocessing import OrdinalEncoder\n';
            addedImports.add('OrdinalEncoder');
          }
          break;
        case 'Missing Values':
          // Add imports based on missing values method
          if (!addedImports.has('SimpleImputer')) {
            generatedCode += 'from sklearn.impute import SimpleImputer\n';
            addedImports.add('SimpleImputer');
          }
          if (config.missingMethod === 'knn' && !addedImports.has('KNNImputer')) {
            generatedCode += 'from sklearn.impute import KNNImputer\n';
            addedImports.add('KNNImputer');
          }
          break;
        case 'Text Processing':
          // Add imports based on text processing method
          if (!addedImports.has('TfidfVectorizer')) {
            generatedCode += 'from sklearn.feature_extraction.text import TfidfVectorizer\n';
            addedImports.add('TfidfVectorizer');
          }
          if (config.textMethod === 'count' && !addedImports.has('CountVectorizer')) {
            generatedCode += 'from sklearn.feature_extraction.text import CountVectorizer\n';
            addedImports.add('CountVectorizer');
          }
          if (config.textMethod === 'hash' && !addedImports.has('HashingVectorizer')) {
            generatedCode += 'from sklearn.feature_extraction.text import HashingVectorizer\n';
            addedImports.add('HashingVectorizer');
          }
          break;
        case 'Dimensionality Reduction':
          // Add imports based on reduction method
          if (!addedImports.has('PCA')) {
            generatedCode += 'from sklearn.decomposition import PCA\n';
            addedImports.add('PCA');
          }
          if (config.reductionMethod === 'svd' && !addedImports.has('TruncatedSVD')) {
            generatedCode += 'from sklearn.decomposition import TruncatedSVD\n';
            addedImports.add('TruncatedSVD');
          }
          if (config.reductionMethod === 'kpca' && !addedImports.has('KernelPCA')) {
            generatedCode += 'from sklearn.decomposition import KernelPCA\n';
            addedImports.add('KernelPCA');
          }
          break;
      }
    });
    
    generatedCode += '\n';
    
    // Track data flow through the pipeline
    let currentDataX = 'X';
    let currentDataY = 'y';
    let hasTrainTestSplit = false;
    
    // Add data loading (if Data Input node exists)
    if (executionPath.some(node => node.data.label === 'Data Input')) {
      generatedCode += '# Load data\n';
      generatedCode += 'data = pd.read_csv("data.csv")\n';
      generatedCode += 'X = data.drop("target", axis=1)\n';
      generatedCode += 'y = data["target"]\n\n';
    }
    
    // Process nodes in execution order with proper data flow
    executionPath.forEach((node, index) => {
      const config = nodeConfigs[node.id] || {};
      const varName = nodeVariables[node.id];
      const prevNode = index > 0 ? executionPath[index - 1] : null;
      const prevVarName = prevNode ? nodeVariables[prevNode.id] : null;
      
      switch (node.data.label) {
      case 'Preprocessing':
          generatedCode += `# Preprocessing - ${varName}\n`;
          
          if (config.scaling === 'standard') {
            generatedCode += `scaler_${varName} = StandardScaler()\n`;
            generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_scaled`;
          } else if (config.scaling === 'minmax') {
            generatedCode += 'from sklearn.preprocessing import MinMaxScaler\n';
            generatedCode += `scaler_${varName} = MinMaxScaler()\n`;
            generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_scaled`;
          } else if (config.scaling === 'robust') {
            generatedCode += 'from sklearn.preprocessing import RobustScaler\n';
            generatedCode += `scaler_${varName} = RobustScaler()\n`;
            generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_scaled`;
          } else if (config.scaling === 'maxabs') {
            generatedCode += 'from sklearn.preprocessing import MaxAbsScaler\n';
            generatedCode += `scaler_${varName} = MaxAbsScaler()\n`;
            generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_scaled`;
          }
          
          // Add feature selection if enabled
          if (config.featureSelection === true) {
            generatedCode += '# Feature Selection\n';
            generatedCode += 'from sklearn.feature_selection import VarianceThreshold\n';
            generatedCode += `selector_${varName} = VarianceThreshold(threshold=0.01)\n`;
            generatedCode += `${varName}_selected = selector_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_selected`;
          }
          
          // Train-test split after preprocessing
          if (!hasTrainTestSplit) {
            generatedCode += `X_train, X_test, y_train, y_test = train_test_split(${currentDataX}, ${currentDataY}, test_size=0.2, random_state=42)\n\n`;
            hasTrainTestSplit = true;
          } else {
            generatedCode += `X_train, X_test, y_train, y_test = train_test_split(${currentDataX}, ${currentDataY}, test_size=0.2, random_state=42)\n\n`;
          }
          break;
          
        case 'Scaling':
          generatedCode += `# Scaling - ${varName}\n`;
          switch (config.scalingMethod) {
            case 'standard':
              generatedCode += `scaler_${varName} = StandardScaler()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
              break;
            case 'minmax':
              generatedCode += 'from sklearn.preprocessing import MinMaxScaler\n';
              generatedCode += `scaler_${varName} = MinMaxScaler()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
              break;
            case 'robust':
              generatedCode += 'from sklearn.preprocessing import RobustScaler\n';
              generatedCode += `scaler_${varName} = RobustScaler()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
              break;
            case 'maxabs':
              generatedCode += 'from sklearn.preprocessing import MaxAbsScaler\n';
              generatedCode += `scaler_${varName} = MaxAbsScaler()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
              break;
            case 'normalizer':
              generatedCode += 'from sklearn.preprocessing import Normalizer\n';
              generatedCode += `scaler_${varName} = Normalizer()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
              break;
            default:
              generatedCode += `scaler_${varName} = StandardScaler()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = scaler_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = scaler_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_scaled = scaler_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_scaled`;
                generatedCode += '\n';
              }
          }
          break;
          
        case 'Encoding':
          generatedCode += `# Encoding - ${varName}\n`;
          switch (config.encodingMethod) {
            case 'onehot':
              generatedCode += 'from sklearn.preprocessing import OneHotEncoder\n';
              generatedCode += `encoder_${varName} = OneHotEncoder()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = encoder_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = encoder_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_encoded = encoder_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_encoded`;
                generatedCode += '\n';
              }
              break;
            case 'label':
              generatedCode += 'from sklearn.preprocessing import LabelEncoder\n';
              generatedCode += `encoder_${varName} = LabelEncoder()\n`;
              generatedCode += `${varName}_encoded = encoder_${varName}.fit_transform(${currentDataX})\n`;
              currentDataX = `${varName}_encoded`;
              generatedCode += '\n';
              break;
            case 'ordinal':
              generatedCode += 'from sklearn.preprocessing import OrdinalEncoder\n';
              generatedCode += `encoder_${varName} = OrdinalEncoder()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = encoder_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = encoder_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_encoded = encoder_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_encoded`;
                generatedCode += '\n';
              }
              break;
            default:
              generatedCode += 'from sklearn.preprocessing import OneHotEncoder\n';
              generatedCode += `encoder_${varName} = OneHotEncoder()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = encoder_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = encoder_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_encoded = encoder_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_encoded`;
                generatedCode += '\n';
              }
          }
          break;
          
        case 'Missing Values':
          generatedCode += `# Missing Values Handling - ${varName}\n`;
          switch (config.missingMethod) {
            case 'mean':
              generatedCode += 'from sklearn.impute import SimpleImputer\n';
              generatedCode += `imputer_${varName} = SimpleImputer(strategy="mean")\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = imputer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = imputer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_imputed = imputer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_imputed`;
                generatedCode += '\n';
              }
              break;
            case 'median':
              generatedCode += 'from sklearn.impute import SimpleImputer\n';
              generatedCode += `imputer_${varName} = SimpleImputer(strategy="median")\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = imputer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = imputer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_imputed = imputer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_imputed`;
                generatedCode += '\n';
              }
              break;
            case 'mode':
              generatedCode += 'from sklearn.impute import SimpleImputer\n';
              generatedCode += `imputer_${varName} = SimpleImputer(strategy="most_frequent")\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = imputer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = imputer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_imputed = imputer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_imputed`;
                generatedCode += '\n';
              }
              break;
            case 'knn':
              generatedCode += 'from sklearn.impute import KNNImputer\n';
              generatedCode += `imputer_${varName} = KNNImputer()\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = imputer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = imputer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_imputed = imputer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_imputed`;
                generatedCode += '\n';
              }
              break;
            case 'drop':
              generatedCode += '# Drop rows with missing values\n';
              generatedCode += `# This should be handled during data loading\n`;
              generatedCode += '\n';
              break;
            default:
              generatedCode += 'from sklearn.impute import SimpleImputer\n';
              generatedCode += `imputer_${varName} = SimpleImputer(strategy="mean")\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = imputer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = imputer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_imputed = imputer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_imputed`;
                generatedCode += '\n';
              }
          }
          break;
          
        case 'Text Processing':
          generatedCode += `# Text Processing - ${varName}\n`;
          switch (config.textMethod) {
            case 'tfidf':
              generatedCode += 'from sklearn.feature_extraction.text import TfidfVectorizer\n';
              generatedCode += `vectorizer_${varName} = TfidfVectorizer()\n`;
              generatedCode += `${varName}_vectorized = vectorizer_${varName}.fit_transform(${currentDataX})\n`;
              currentDataX = `${varName}_vectorized`;
              generatedCode += '\n';
              break;
            case 'count':
              generatedCode += 'from sklearn.feature_extraction.text import CountVectorizer\n';
              generatedCode += `vectorizer_${varName} = CountVectorizer()\n`;
              generatedCode += `${varName}_vectorized = vectorizer_${varName}.fit_transform(${currentDataX})\n`;
              currentDataX = `${varName}_vectorized`;
              generatedCode += '\n';
              break;
            case 'hash':
              generatedCode += 'from sklearn.feature_extraction.text import HashingVectorizer\n';
              generatedCode += `vectorizer_${varName} = HashingVectorizer()\n`;
              generatedCode += `${varName}_vectorized = vectorizer_${varName}.fit_transform(${currentDataX})\n`;
              currentDataX = `${varName}_vectorized`;
              generatedCode += '\n';
              break;
            default:
              generatedCode += 'from sklearn.feature_extraction.text import TfidfVectorizer\n';
              generatedCode += `vectorizer_${varName} = TfidfVectorizer()\n`;
              generatedCode += `${varName}_vectorized = vectorizer_${varName}.fit_transform(${currentDataX})\n`;
              currentDataX = `${varName}_vectorized`;
              generatedCode += '\n';
          }
          break;
          
        case 'Dimensionality Reduction':
          generatedCode += `# Dimensionality Reduction - ${varName}\n`;
          switch (config.reductionMethod) {
            case 'pca':
              generatedCode += 'from sklearn.decomposition import PCA\n';
              generatedCode += `reducer_${varName} = PCA(n_components=10)\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = reducer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = reducer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_reduced = reducer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_reduced`;
                generatedCode += '\n';
              }
              break;
            case 'svd':
              generatedCode += 'from sklearn.decomposition import TruncatedSVD\n';
              generatedCode += `reducer_${varName} = TruncatedSVD(n_components=10)\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = reducer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = reducer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_reduced = reducer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_reduced`;
                generatedCode += '\n';
              }
              break;
            case 'kpca':
              generatedCode += 'from sklearn.decomposition import KernelPCA\n';
              generatedCode += `reducer_${varName} = KernelPCA(n_components=10)\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = reducer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = reducer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_reduced = reducer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_reduced`;
                generatedCode += '\n';
              }
              break;
            default:
              generatedCode += 'from sklearn.decomposition import PCA\n';
              generatedCode += `reducer_${varName} = PCA(n_components=10)\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = reducer_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = reducer_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_reduced = reducer_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_reduced`;
                generatedCode += '\n';
              }
          }
          break;
          
        case 'PCA':
          generatedCode += `# PCA - ${varName}\n`;
          generatedCode += `pca_${varName} = PCA(`;
          const pcaParams = [];
          if (config.nComponents) pcaParams.push(`n_components=${config.nComponents}`);
          if (config.whiten === true) pcaParams.push('whiten=True');
          generatedCode += pcaParams.join(', ') + ')\n';
          if (hasTrainTestSplit) {
            generatedCode += `X_train = pca_${varName}.fit_transform(X_train)\n`;
            generatedCode += `X_test = pca_${varName}.transform(X_test)\n\n`;
          } else {
            generatedCode += `${varName}_pca = pca_${varName}.fit_transform(${currentDataX})\n`;
            currentDataX = `${varName}_pca`;
            generatedCode += '\n';
          }
          break;
          
        case 'Feature Selection':
          generatedCode += `# Feature Selection - ${varName}\n`;
          switch (config.method) {
            case 'variance':
              generatedCode += `selector_${varName} = VarianceThreshold(threshold=${config.threshold || 0.01})\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = selector_${varName}.fit_transform(X_train)\n`;
                generatedCode += `X_test = selector_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_selected = selector_${varName}.fit_transform(${currentDataX})\n`;
                currentDataX = `${varName}_selected`;
                generatedCode += '\n';
              }
              break;
            case 'univariate':
              generatedCode += `selector_${varName} = SelectKBest(score_func=f_classif, k=10)\n`;
              if (hasTrainTestSplit) {
                generatedCode += `X_train = selector_${varName}.fit_transform(X_train, y_train)\n`;
                generatedCode += `X_test = selector_${varName}.transform(X_test)\n\n`;
              } else {
                generatedCode += `${varName}_selected = selector_${varName}.fit_transform(${currentDataX}, ${currentDataY})\n`;
                currentDataX = `${varName}_selected`;
                generatedCode += '\n';
              }
              break;
            case 'recursive':
              // RFE requires a model, will be handled with model nodes
              break;
          }
          break;
          
        case 'Linear Regression':
        case 'Logistic Regression':
        case 'Decision Tree':
        case 'Random Forest':
        case 'SVM':
        case 'K-Means':
          // Ensure we have train/test split for supervised models
          if (!hasTrainTestSplit && node.data.label !== 'K-Means') {
            generatedCode += `X_train, X_test, y_train, y_test = train_test_split(${currentDataX}, ${currentDataY}, test_size=0.2, random_state=42)\n\n`;
            hasTrainTestSplit = true;
          }
          
          // Model training
          generatedCode += `# ${node.data.label} - ${varName}\n`;
          
          switch (node.data.label) {
            case 'Linear Regression':
              generatedCode += `${varName} = LinearRegression(`;
              const lrParams = [];
              if (config.fitIntercept === false) lrParams.push('fit_intercept=False');
              if (config.normalize === true) lrParams.push('normalize=True');
              generatedCode += lrParams.join(', ') + ')\n';
              if (node.data.label === 'K-Means') {
                generatedCode += `${varName}.fit(${currentDataX})\n\n`;
              } else {
                generatedCode += `${varName}.fit(X_train, y_train)\n\n`;
              }
              break;
              
            case 'Logistic Regression':
              generatedCode += `${varName} = LogisticRegression(`;
              const logrParams = [];
              if (config.penalty && config.penalty !== 'l2') logrParams.push(`penalty='${config.penalty}'`);
              if (config.solver && config.solver !== 'lbfgs') logrParams.push(`solver='${config.solver}'`);
              if (config.maxIter && config.maxIter !== 100) logrParams.push(`max_iter=${config.maxIter}`);
              generatedCode += logrParams.join(', ') + ')\n';
              generatedCode += `${varName}.fit(X_train, y_train)\n\n`;
              break;
              
            case 'Decision Tree':
              generatedCode += `${varName} = DecisionTreeClassifier(`;
              const dtParams = [];
              if (config.criterion && config.criterion !== 'gini') dtParams.push(`criterion='${config.criterion}'`);
              if (config.maxDepth) dtParams.push(`max_depth=${config.maxDepth}`);
              if (config.minSamplesSplit && config.minSamplesSplit !== 2) dtParams.push(`min_samples_split=${config.minSamplesSplit}`);
              generatedCode += dtParams.join(', ') + ')\n';
              generatedCode += `${varName}.fit(X_train, y_train)\n\n`;
              break;
              
            case 'Random Forest':
              generatedCode += `${varName} = RandomForestClassifier(`;
              const rfParams = [];
              if (config.nEstimators && config.nEstimators !== 100) rfParams.push(`n_estimators=${config.nEstimators}`);
              if (config.criterion && config.criterion !== 'gini') rfParams.push(`criterion='${config.criterion}'`);
              if (config.maxDepth) rfParams.push(`max_depth=${config.maxDepth}`);
              generatedCode += rfParams.join(', ') + ')\n';
              generatedCode += `${varName}.fit(X_train, y_train)\n\n`;
              break;
              
            case 'SVM':
              generatedCode += `${varName} = SVC(`;
              const svmParams = [];
              if (config.kernel && config.kernel !== 'rbf') svmParams.push(`kernel='${config.kernel}'`);
              if (config.C && config.C !== 1.0) svmParams.push(`C=${config.C}`);
              if (config.gamma && config.gamma !== 'scale') svmParams.push(`gamma='${config.gamma}'`);
              generatedCode += svmParams.join(', ') + ')\n';
              generatedCode += `${varName}.fit(X_train, y_train)\n\n`;
              break;
              
            case 'K-Means':
              generatedCode += `${varName} = KMeans(`;
              const kmParams = [];
              if (config.nClusters && config.nClusters !== 8) kmParams.push(`n_clusters=${config.nClusters}`);
              if (config.init && config.init !== 'k-means++') kmParams.push(`init='${config.init}'`);
              if (config.maxIter && config.maxIter !== 300) kmParams.push(`max_iter=${config.maxIter}`);
              generatedCode += kmParams.join(', ') + ')\n';
              generatedCode += `${varName}.fit(${currentDataX})\n\n`;
              break;
          }
          break;
          
        case 'Grid Search':
          generatedCode += `# Grid Search - ${varName}\n`;
          // Find the previous model node to apply grid search to
          const modelNode = executionPath[executionPath.indexOf(node) - 1];
          if (modelNode) {
            const modelVarName = nodeVariables[modelNode.id];
            generatedCode += `param_grid_${varName} = {}\n`;
            generatedCode += `${varName} = GridSearchCV(${modelVarName}, param_grid_${varName}, cv=5)\n`;
            generatedCode += `${varName}.fit(X_train, y_train)\n`;
            generatedCode += `${modelVarName} = ${varName}.best_estimator_\n\n`;
          }
          break;
          
        case 'Random Search':
          generatedCode += `# Random Search - ${varName}\n`;
          // Find the previous model node to apply random search to
          const randModelNode = executionPath[executionPath.indexOf(node) - 1];
          if (randModelNode) {
            const modelVarName = nodeVariables[randModelNode.id];
            generatedCode += `param_dist_${varName} = {}\n`;
            generatedCode += `${varName} = RandomizedSearchCV(${modelVarName}, param_dist_${varName}, cv=5)\n`;
            generatedCode += `${varName}.fit(X_train, y_train)\n`;
            generatedCode += `${modelVarName} = ${varName}.best_estimator_\n\n`;
          }
          break;
          
        case 'Model Evaluation':
          generatedCode += `# Model Evaluation - ${varName}\n`;
          // Find the previous model node to evaluate
          const evalModelNode = executionPath[executionPath.indexOf(node) - 1];
          if (evalModelNode) {
            const modelVarName = nodeVariables[evalModelNode.id];
            generatedCode += `score_${varName} = ${modelVarName}.score(X_test, y_test)\n`;
            generatedCode += `print(f"${modelVarName} accuracy: {score_${varName}}")\n`;
            
            if (config.metrics && config.metrics.length > 0) {
              if (config.metrics.includes('precision')) {
                generatedCode += 'from sklearn.metrics import precision_score\n';
                generatedCode += `precision_${varName} = precision_score(y_test, ${modelVarName}.predict(X_test))\n`;
                generatedCode += `print(f"${modelVarName} precision: {precision_${varName}}")\n`;
              }
              if (config.metrics.includes('recall')) {
                generatedCode += 'from sklearn.metrics import recall_score\n';
                generatedCode += `recall_${varName} = recall_score(y_test, ${modelVarName}.predict(X_test))\n`;
                generatedCode += `print(f"${modelVarName} recall: {recall_${varName}}")\n`;
              }
              if (config.metrics.includes('f1')) {
                generatedCode += 'from sklearn.metrics import f1_score\n';
                generatedCode += `f1_${varName} = f1_score(y_test, ${modelVarName}.predict(X_test))\n`;
                generatedCode += `print(f"${modelVarName} F1 Score: {f1_${varName}}")\n`;
              }
            }
          }
          break;
          
        case 'Model Comparison':
          generatedCode += `# Model Comparison - ${varName}\n`;
          // This would typically collect metrics from multiple models and compare them
          // For now, we'll add a placeholder for the comparison logic
          generatedCode += `# Model comparison logic would go here\n`;
          generatedCode += `# This node would collect metrics from multiple models and compare them\n`;
          generatedCode += `# Visualization type: ${config.visualization || 'bar'}\n`;
          if (config.statisticalTest) {
            generatedCode += `# Include statistical significance test\n`;
          }
          generatedCode += '\n';
          break;
      }
    });
    
    
    setCode(generatedCode);
  }, [nodes, edges, nodeConfigs]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_id', projectName);
    
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      if (response.ok) {
        setResults({ message: `File uploaded successfully: ${result.shape[0]} rows, ${result.shape[1]} columns` });
      } else {
        setResults({ error: result.error });
      }
    } catch (error) {
      setResults({ error: 'Failed to upload file' });
    }
  };

  const executePipeline = async () => {
    try {
      const response = await fetch('http://localhost:5000/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          code: code,
          dataset_id: projectName
        })
      });
      
      const result = await response.json();
      if (response.ok) {
        setResults(result);
        if (result.plot) {
          setPlot(result.plot);
        }
      } else {
        setResults({ error: result.error });
      }
    } catch (error) {
      setResults({ error: 'Failed to execute pipeline' });
    }
  };

  const saveProject = async () => {
    try {
      const response = await fetch('http://localhost:5000/save-project', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          project_id: projectName,
          project_data: {
            nodes,
            edges,
            nodeConfigs
          }
        })
      });
      
      const result = await response.json();
      if (response.ok) {
        setResults({ message: 'Project saved successfully' });
      } else {
        setResults({ error: result.error });
      }
    } catch (error) {
      setResults({ error: 'Failed to save project' });
    }
  };

  const loadProject = async () => {
    try {
      const response = await fetch(`http://localhost:5000/load-project/${projectName}`);
      
      if (response.ok) {
        const projectData = await response.json();
        setNodes(projectData.nodes || []);
        setEdges(projectData.edges || []);
        setNodeConfigs(projectData.nodeConfigs || {});
        setResults({ message: 'Project loaded successfully' });
      } else {
        setResults({ error: 'Project not found' });
      }
    } catch (error) {
      setResults({ error: 'Failed to load project' });
    }
  };

  const exportCode = async () => {
    try {
      const response = await fetch('http://localhost:5000/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          code: code,
          filename: `${projectName}.py`
        })
      });
      
      const result = await response.json();
      if (response.ok) {
        // Create a download link
        const blob = new Blob([result.code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        setResults({ message: 'Code exported successfully' });
      } else {
        setResults({ error: result.error });
      }
    } catch (error) {
      setResults({ error: 'Failed to export code' });
    }
  };

  const validatePipeline = async () => {
    try {
      const response = await fetch('http://localhost:5000/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          nodes,
          edges
        })
      });
      
      const result = await response.json();
      if (response.ok) {
        setResults({ message: result.message, valid: result.valid });
      } else {
        setResults({ error: result.error });
      }
    } catch (error) {
      setResults({ error: 'Failed to validate pipeline' });
    }
  };

  return (
    <div className="app">
      <div className="node-panel-container">
        <NodePanel onAddNode={addNode} />
        <div className="project-controls">
          <input 
            type="text" 
            value={projectName} 
            onChange={(e) => setProjectName(e.target.value)} 
            placeholder="Project Name"
            className="project-name-input"
          />
          <button onClick={() => fileInputRef.current.click()} className="control-button">
            Upload Data
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            style={{ display: 'none' }} 
            accept=".csv,.json"
          />
          <button onClick={executePipeline} className="control-button">
            Execute Pipeline
          </button>
          <button onClick={validatePipeline} className="control-button">
            Validate Pipeline
          </button>
          <button onClick={saveProject} className="control-button">
            Save Project
          </button>
          <button onClick={loadProject} className="control-button">
            Load Project
          </button>
          <button onClick={exportCode} className="control-button">
            Export Code
          </button>
        </div>
      </div>
      <div className="main-content">
<ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={12} size={1} />
        </ReactFlow>
      </div>
      <div 
        className="resize-divider" 
        onMouseDown={handleResizeMouseDown}
      />
      <div className="right-panel" style={{ width: rightPanelWidth }}>
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
        <div className="results-panel">
          <h3>Results</h3>
          {results && (
            <div className="results-content">
              {results.message && <p className="result-message">{results.message}</p>}
              {results.error && <p className="result-error">{results.error}</p>}
              {results.model_type && <p>Model: {results.model_type}</p>}
              {plot && (
                <div className="plot-container">
                  <img src={`data:image/png;base64,${plot}`} alt="Model visualization" />
                </div>
              )}
            </div>
          )}
        </div>
        <div className="code-panel">
          <h3>Generated Code</h3>
          <div className="code-display">
            <SyntaxHighlighter language="python" style={tomorrow} showLineNumbers>
              {code}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
