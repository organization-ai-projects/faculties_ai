import axios from 'axios';
import * as fs from 'fs';

const generateStructure = async (prompt: string) => {
    const response = await axios.post('http://localhost:8000/generate', { prompt });
    return response.data;
};

const createProject = async (projectName: string) => {
    const prompt = `Créer une structure complète pour le projet nommé ${projectName}.`;
    const output = await generateStructure(prompt);

    fs.mkdirSync(`projects/${projectName}`, { recursive: true });
    fs.writeFileSync(`projects/${projectName}/README.md`, output.files['README.md']);
    console.log(`Structure générée pour ${projectName}`);
};

createProject('example-project');
