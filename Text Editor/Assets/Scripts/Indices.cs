public class Indices
{
    public int row;
    public int col;

    /// <summary>
    /// Checks if the current index is before the other one.
    /// 
    /// Returns false if both are the same.
    /// </summary>
    /// <param name="other">The other index</param>
    /// <returns>True if current is before the other one</returns>
    public bool IsBefore(Indices other)
    {
        if (row < other.row)
            return true;
        else if (row == other.row && col < other.col)
            return true;
        else
            return false;
    }
}
