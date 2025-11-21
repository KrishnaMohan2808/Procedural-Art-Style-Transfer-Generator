import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultViewer from "./components/ResultViewer";
import "./styles/App.css";

function App() {
  const [resultUrl, setResultUrl] = useState(null);

  return (
    <div className="app">
      <UploadForm onResult={setResultUrl} />
      <ResultViewer resultUrl={resultUrl} />
    </div>
  );
}

export default App;
