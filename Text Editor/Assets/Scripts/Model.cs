using UnityEngine;
using UnityEngine.Assertions;

public class Model {

    // TODO: automatically detect model size based on canvas and font size
    private char[,] model;
    public int max_rows;
    public int max_columns;

    public Model(string text, int max_rows, int max_columns)
    {
        this.max_rows = max_rows;
        this.max_columns = max_columns;
        BuildModel(text);
    }

    public void BuildModel(string text)
    {
        model = new char[max_rows, max_columns];
        string[] lines = text.Split(new char[] { '\n' });

        // TODO: Assume for now, that text will always initially be within model size. Fix later
        Assert.IsTrue(lines.Length <= model.GetLength(0), "Too many lines in text");

        for (int i = 0; i < lines.Length; i++)
        {
            Assert.IsTrue(lines[i].Length <= model.GetLength(1), string.Format("Too many characters in line {0}", i));
            for (int j = 0; j < lines[i].Length; j++)
            {
                model[i, j] = lines[i][j];
            }
        }
    }

    public char GetLetter(Indices c)
    {
        if (c.row < model.GetLength(0) && c.col < model.GetLength(1))
            return model[c.row, c.col];
        else
            return '\0';
    }

    public string GetWord(Indices c)
    {
        if ((c.row >= model.GetLength(0) && c.col >= model.GetLength(1)) || model[c.row, c.col] == ' ' || model[c.row, c.col] == '\0')
            return "";

        Indices beg = new Indices{ row = c.row, col = 0 };
 
        // Find the beginning of the string (words are only on the same line)
        for (int i=c.col; i>=0; i--)
        {
            if (model[c.row, i] == ' ')
            {
                beg.col = i + 1;
                break;
            }

        }

        // Keep reading until end of line or end of word (space)
        string word = "";
        for (int i = beg.col; i < model.GetLength(1); i++)
        {
            if (model[beg.row, i] == ' ' || model[beg.row, i] == '\0')
                break;

            word += model[beg.row, i];
        }

        return word;
    }

    public string GetSelection(Indices start, Indices end)
    {
        if (start.row >= model.GetLength(0) && start.col >= model.GetLength(1))
            return "";

        if (end.IsBefore(start))
        {
            Indices temp = start;
            start = end;
            end = temp;
        }
        
        string selection = "";
        int line_start_col = start.col;
        for (int i = start.row; i < Mathf.Min(end.row+1, model.GetLength(0)); i++)
        {
            for (int j = line_start_col; j < model.GetLength(1); j++)
            {
                if (model[i,j] != '\0')
                    selection += model[i, j];
                if (i == end.row && j == end.col)
                {
                    i = model.GetLength(0); // Trick to break out of the outer loop
                    break;
                }
            }
            selection += '\n';
            line_start_col = 0;
        }

        return selection;
    }
 
}
