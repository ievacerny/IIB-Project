using UnityEngine;

public class ModelViewMapping {

    private float margin;
    private float width;
    private float height;

    public ModelViewMapping(float margin, float width, float height)
    {
        this.margin = margin;
        this.width = width;
        this.height = height;
    }

    public Indices CoordsToIndex(Vector3 coords)
    {
        var indices = new Indices();

        indices.row = (int) ((coords.y - 0.5f) / -height);
        indices.col = (int) ((coords.x + (0.5f - margin)) / width);
        //Debug.Log(string.Format("Row {0}, column {1}", indices.row, indices.col));

        return indices;
    }

    public Vector3 IndexToCoords(Indices indices)
    {
        var coords = new Vector3();

        coords.y = ((float)indices.row + 0.5f) * -height + 0.5f;
        coords.x = ((float)indices.col + 0.5f) * width - (0.5f - margin);
        coords.z = 0f;

        return coords;
    }




    Vector3 end = new Vector3(0.4818f, -0.4326f, 0f);
    Vector3 beg = new Vector3(-0.4874f, 0.4062f, 0f);

    // Function used to optimise the width and height of the characters, and the margin of the page
    void OptimiseParams()
    {
        float best_margin = 100f;
        float best_width = 100f;
        float min_distance = 100f;
        for (margin = 0f; margin <= 0.007f; margin += 0.00001f)
        {
            for (width = 0.01f; width <= 0.06f; width += 0.0001f)
            {
                var end_i = CoordsToIndex(end);
                var end_c = IndexToCoords(end_i);

                var beg_i = CoordsToIndex(beg);
                var beg_c = IndexToCoords(beg_i);

                if ((Mathf.Abs(end_c.x - end.x) + Mathf.Abs(beg_c.x - beg.x) < min_distance) &&
                    end_i.col == 52 && beg_i.col == 0)
                {
                    min_distance = Mathf.Abs(end_c.y - end.y) + Mathf.Abs(beg_c.y - beg.y);
                    best_margin = margin;
                    best_width = width;
                }
            }
        }

        Debug.Log(min_distance);
        Debug.Log(best_width);
        Debug.Log(best_margin);
        margin = best_margin;
        width = best_width;
    }

}


public class Indices
{
    public int row;
    public int col;
}
