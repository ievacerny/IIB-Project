using System;
using System.Net.Sockets;
using UnityEngine;

class TCPClient
{
    #region Private Attributes
    private readonly string serverIpAddressString = "localhost";
    private readonly int port = 65432;
    private TcpClient client;
    #endregion

    #region Constructor, Destructor

    /// <summary>
    /// TCPClient class constructor. Establishes the connections with an open server.
    /// </summary>
    /// <returns></returns>
    public TCPClient()
    {
        client = Connect(serverIpAddressString, port);
        if (!client.Connected)
        {
            Debug.LogError(String.Format("Failed to connect to {0}:{1}", serverIpAddressString, port));
            return;
        }
    }

    /// <summary>
    /// TCPClient class destructor. Closes the socket.
    /// </summary>
    /// <returns></returns>
    ~TCPClient()
    {
        Debug.Log("Closing the socket");
        try
        {
            client.Close();
        }
        catch (Exception e)
        {
            Debug.Log(e.Message);
        }
    }

    #endregion

    #region Private Methods

    private TcpClient Connect(String server, int port)
    {
        TcpClient client = new TcpClient();

        try
        {
            client = new TcpClient(server, port);
        }
        catch (ArgumentNullException e)
        {
            Console.WriteLine("ArgumentNullException: {0}", e);
        }
        catch (SocketException e)
        {
            Console.WriteLine("SocketException: {0}", e);
        }

        return client;
    }
    
    #endregion

    #region Public Interface Methods

    /// <summary>
    /// Send a message to the server.
    /// </summary>
    /// <param name="message"></param>
    /// <param name="log">[default=false] True if sent message should be logged.</param>
    /// <returns></returns>
    public void SendMessage(string message, bool log=false)
    {
        // Translate the passed message into ASCII and store it as a Byte array.
        Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);
     
        // Get a client stream for writing.
        NetworkStream stream = client.GetStream();
        
        // Send the message to the connected TcpServer. 
        stream.Write(data, 0, data.Length);
        
        // Report what was sent to console
        if (log)
            Debug.Log("Sent: " + message);

        // Flush the stream
        stream.Flush();
    }

    /// <summary>
    /// Listen for and read a message from the server (blocking call).
    /// </summary>
    /// <param name="log">[default=false] True if received message should be logged.</param>
    /// <returns>The message received.</returns>
    public string ReadMessage(bool log=false)
    {
        // Buffer to store the response bytes.
        Byte[] data = new Byte[1024];
        
        // String to store the response ASCII representation.
        String responseData = String.Empty;
        
        // Get a client stream for writing.
        NetworkStream stream = client.GetStream();
        
        // Read the first batch of the TcpServer response bytes.
        Int32 bytes = stream.Read(data, 0, data.Length);
        responseData = System.Text.Encoding.ASCII.GetString(data, 0, bytes);
        
        // Report what was received to console
        if (log)
            Debug.Log("Received: " + responseData);

        return responseData;
    }

    /// <summary>
    /// Check the connectivity status
    /// </summary>
    /// <returns>Returns true/false.</returns>
    public bool IsClientConnected()
    {
        return client.Connected;
    }

    #endregion
}
