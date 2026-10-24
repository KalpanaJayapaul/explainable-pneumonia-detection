import React, {useState} from "react";
import {createRoot} from "react-dom/client";
import {
  AppBar, Toolbar, Typography, Container, Card, CardContent, Button,
  Box, Alert, Chip, LinearProgress, Stack
} from "@mui/material";

const API = "http://127.0.0.1:8000/api/predict/";

function App(){
  const [file,setFile] = useState(null);
  const [preview,setPreview] = useState("");
  const [result,setResult] = useState(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState("");

  const choose = e => {
    const f=e.target.files?.[0];
    if(!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null); setError("");
  };

  const predict = async () => {
    if(!file){setError("Please select a chest X-ray image.");return;}
    setLoading(true); setError(""); setResult(null);
    try{
      const form=new FormData();
      form.append("image",file);
      const res=await fetch(API,{method:"POST",body:form});
      const data=await res.json();
      if(!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    }catch(e){setError(e.message)}
    finally{setLoading(false)}
  };

  return <Box sx={{minHeight:"100vh",background:"#f5f7fb"}}>
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6">Pneumonia XAI Research System</Typography>
      </Toolbar>
    </AppBar>
    <Container maxWidth="md" sx={{py:5}}>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        Explainable Pneumonia Detection
      </Typography>
      <Typography color="text.secondary" sx={{mb:3}}>
        Upload a chest X-ray to obtain a model prediction, model-derived risk
        category and Grad-CAM explanation.
      </Typography>

      <Card>
        <CardContent>
          <Stack spacing={2}>
            <Button variant="outlined" component="label">
              Select Chest X-Ray
              <input hidden type="file" accept="image/*" onChange={choose}/>
            </Button>
            {preview && <Box component="img" src={preview}
              sx={{maxHeight:350,maxWidth:"100%",objectFit:"contain",mx:"auto"}}/>}
            {file && <Typography>{file.name}</Typography>}
            <Button variant="contained" onClick={predict} disabled={loading}>
              Analyze Image
            </Button>
            {loading && <LinearProgress/>}
            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </CardContent>
      </Card>

      {result && <Card sx={{mt:3}}>
        <CardContent>
          <Typography variant="h5" gutterBottom>Result</Typography>
          <Stack direction="row" spacing={1} sx={{mb:2}}>
            <Chip label={result.prediction}
              color={result.prediction==="PNEUMONIA"?"error":"success"}/>
            <Chip label={`Risk: ${result.risk_category}`}/>
          </Stack>

          <Typography>
            Pneumonia probability: {(result.pneumonia_probability*100).toFixed(2)}%
          </Typography>
          <Typography>
            Model confidence: {(result.confidence*100).toFixed(2)}%
          </Typography>

          {result.gradcam_url &&
            <Box sx={{mt:3}}>
              <Typography variant="h6">Grad-CAM Explanation</Typography>
              <Box component="img"
                src={`http://127.0.0.1:8000${result.gradcam_url}`}
                sx={{width:"100%",maxHeight:500,objectFit:"contain"}}/>
            </Box>
          }

          <Alert severity="warning" sx={{mt:3}}>
            {result.disclaimer}
          </Alert>
        </CardContent>
      </Card>}
    </Container>
  </Box>
}
createRoot(document.getElementById("root")).render(<App/>);
