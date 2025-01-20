export interface Generator {
    generate(prompt: string): Promise<string>;
}