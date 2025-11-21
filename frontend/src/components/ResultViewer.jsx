import React from "react";

const ResultViewer = ({ resultUrl }) => {
  if (!resultUrl) return null;

  return (
    <div className="result">
      <h3>🖼️ Stylized Output</h3>
      <img src={resultUrl} alt="Stylized Result" width="512" height="512" />
    </div>
  );
};

export default ResultViewer;
