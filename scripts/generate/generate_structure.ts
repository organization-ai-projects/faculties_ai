import * as fs from 'fs';

const createStructure = (name: string) => {
    fs.mkdirSync(name, { recursive: true });
    fs.writeFileSync(`${name}/README.md`, '# New Project\n\nCe projet a été généré automatiquement.');
    console.log(`Structure créée pour ${name}`);
};

createStructure('generated-project');
