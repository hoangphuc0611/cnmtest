import React from "react";
import "./main.css";
import ReactLoading from "react-loading";

class Main extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      imageURL:
        "https://www.generationsforpeace.org/wp-content/uploads/2018/03/empty.jpg",
      check: false,
      loading: false,
    };
    this.handleUploadImage = this.handleUploadImage.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }
  handleUploadImage(e) {
    e.preventDefault();
    this.setState({ loading: true });
    const data = new FormData();
    data.append("file", this.uploadInput.files[0]);
    console.log(data);
    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: data,
    })
      .then((response) => response.blob())
      .then((images) => {
        const image = URL.createObjectURL(images);
        this.setState({ imageURL: image });
        this.setState({ loading: false });
      })
      .catch((error) => {
        console.error(error);
      });
  }
  handleChange(event) {
    this.setState({
      imageURL: URL.createObjectURL(event.target.files[0]),
    });
    this.setState({ check: true });
  }
  render() {
    return (
      <form onSubmit={this.handleUploadImage}>
        <div className="image">
          {this.state.loading ? (
            <ReactLoading color="#34abeb" type="balls" />
          ) : (
            <img src={this.state.imageURL} alt="img" />
          )}
        </div>
        <div className="input">
          <input
            ref={(ref) => {
              this.uploadInput = ref;
            }}
            onChange={this.handleChange}
            type="file"
          />
        </div>
        <div>{this.state.check && <button>Predict</button>}</div>
      </form>
    );
  }
}
export default Main;
