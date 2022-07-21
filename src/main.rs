mod log;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use tracing::{error, info};
use vulkano::{
  buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
  command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents,
  },
  device::{
    physical::{PhysicalDevice, PhysicalDeviceType},
    Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
  },
  image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
  impl_vertex,
  instance::{Instance, InstanceCreateInfo},
  pipeline::{
    graphics::{
      input_assembly::InputAssemblyState,
      vertex_input::BuffersDefinition,
      viewport::{Viewport, ViewportState},
    },
    GraphicsPipeline,
  },
  render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
  swapchain::{
    acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
  },
  sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
  event::{Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::{Window, WindowBuilder},
};

fn main() -> anyhow::Result<()> {
  self::log::init();
  run()
}

fn run() -> anyhow::Result<()> {
  let require_extensions = vulkano_win::required_extensions();
  // 第一步，实例和物理设备选择
  // vulkan instance是vulkan一切API的入口点
  // 创建VkInstace后，可以查询VK支持的硬件，
  // 选中其中一个或多个PhysicalDevice进行操作
  let instance = Instance::new(InstanceCreateInfo {
    application_name: Some(env!("CARGO_PKG_NAME").to_owned()),
    enabled_extensions: require_extensions,
    ..Default::default()
  })?;

  let event_loop = EventLoop::new();

  let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone())?;

  let device_extensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
  };
  // 通过查询设备属性选择一个合适的PhysicalDevice
  let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
    .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
    .filter_map(|p| {
      p.queue_families()
        .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
        .map(|q| (p, q))
    })
    .min_by_key(|(p, _)| match p.properties().device_type {
      PhysicalDeviceType::DiscreteGpu => 0,
      PhysicalDeviceType::VirtualGpu => 1,
      PhysicalDeviceType::IntegratedGpu => 2,
      PhysicalDeviceType::Cpu => 3,
      PhysicalDeviceType::Other => 4,
    })
    .unwrap();
  info!(
    "Using device: {} (type: {:?})",
    physical_device.properties().device_name,
    physical_device.properties().device_type,
  );
  // 第二步，逻辑设备与队列
  // 1. 使用更详细的PhysicalDevice特性创建一个逻辑设备
  // 2. 指定队列族，VK将绘制指令，内存操作提交到队列中，异步执行
  let (device, mut queues) = Device::new(
    physical_device,
    DeviceCreateInfo {
      enabled_extensions: physical_device
        .required_extensions()
        .union(&device_extensions),
      queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
      ..Default::default()
    },
  )?;
  let queue = queues.next().ok_or(anyhow::anyhow!("No available queue"))?;

  // 第三步，交换链与窗口表面
  // 我们需要创建一个窗口来显示渲染的图像
  // 交换链是一个渲染目标集合。它可以保证我们正在渲染的图像和当前屏幕图像是两个不同的图像。
  // 这可以确保显示出来的图像是完整的。每次绘制一帧时,可以请求交换链提供一张图像。
  // 绘制完成后,图像被返回到交换链中,在之后某个时刻,图像被显示到屏幕上。
  let (mut swapchain, images) = {
    let surface_capabilities =
      physical_device.surface_capabilities(&surface, Default::default())?;
    let image_format = Some(physical_device.surface_formats(&surface, Default::default())?[0].0);
    Swapchain::new(
      device.clone(),
      surface.clone(),
      vulkano::swapchain::SwapchainCreateInfo {
        min_image_count: surface_capabilities.min_image_count,
        image_format,
        image_usage: ImageUsage::color_attachment(),
        composite_alpha: surface_capabilities
          .supported_composite_alpha
          .iter()
          .next()
          .expect("No supported CompositeAlpha"),
        ..Default::default()
      },
    )?
  };
  #[repr(C)]
  #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
  struct Vertex {
    position: [f32; 2],
  }
  impl_vertex!(Vertex, position);

  let vertices = [
    Vertex {
      position: [-0.5, 0.5],
    },
    Vertex {
      position: [0.0, 0.5],
    },
    Vertex {
      position: [0.25, -0.1],
    },
  ];
  let vertex_buffer =
    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices)?;
  mod vs {
    vulkano_shaders::shader! {
      ty: "vertex",
      src: "
      #version 450

      layout(location = 0) in vec2 position;
      void main() {
        gl_Position = vec4(position, 0.0, 1.0);
      }
      "
    }
  }
  mod fs {
    vulkano_shaders::shader! {
      ty: "fragment",
      src: "
      #version 450

      layout(location = 0) out vec4 f_color;

      void main() {
        f_color = vec4(1.0,0.0,0.0,1.0);
      }
      "
    }
  }
  let vs = vs::load(device.clone())?;
  let fs = fs::load(device.clone())?;

  // 渲染流程
  // 渲染流程描述了渲染操作使用的图像类型,图像的使用方式,图像的内容如何处理。
  let render_pass = vulkano::single_pass_renderpass!(
    device.clone(),
    attachments: {
      color: {
        load: Clear,
        store: Store,
        format: swapchain.image_format(),
        samples: 1,
      }
    },
    pass: {
      color: [color],
      depth_stencil: {}
    }
  )?;

  // 图形管线
  // 图形管线描述了显卡的可配置状态, 以及使用着色器的可编程状态
  let pipeline = GraphicsPipeline::start()
    .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
    .vertex_shader(vs.entry_point("main").unwrap(), ())
    .input_assembly_state(InputAssemblyState::new())
    .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
    .fragment_shader(fs.entry_point("main").unwrap(), ())
    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
    .build(device.clone())?;

  let mut viewport = Viewport {
    origin: [0.0, 0.0],
    dimensions: [0.0, 0.0],
    depth_range: 0.0..1.0,
  };

  // 帧缓冲
  // 从交换链获取图像后,还不能直接在图像上进行绘制,需要将图像包装进帧缓冲中去。
  let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

  let mut recreate_swapchain = false;
  let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

  event_loop.run(move |event, _, control_flow| match event {
    Event::WindowEvent {
      event: WindowEvent::CloseRequested,
      ..
    } => {
      *control_flow = ControlFlow::Exit;
    }
    Event::WindowEvent {
      event: WindowEvent::Resized(_),
      ..
    } => {
      recreate_swapchain = true;
    }
    Event::RedrawEventsCleared => {
      let dimensions = surface.window().inner_size();
      if dimensions.width == 0 || dimensions.height == 0 {
        return;
      }
      previous_frame_end.as_mut().unwrap().cleanup_finished();
      if recreate_swapchain {
        let (new_swapchain, new_image) = match swapchain.recreate(SwapchainCreateInfo {
          image_extent: dimensions.into(),
          ..swapchain.create_info()
        }) {
          Ok(r) => r,
          Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
          Err(e) => panic!("Error creating swapchain: {}", e),
        };
        swapchain = new_swapchain;
        framebuffers = window_size_dependent_setup(&new_image, render_pass.clone(), &mut viewport);
        recreate_swapchain = false;
      }

      let (image_num, suboptimal, acquire_future) =
        match acquire_next_image(swapchain.clone(), None) {
          Ok(r) => r,
          Err(AcquireError::OutOfDate) => {
            recreate_swapchain = true;
            return;
          }
          Err(e) => panic!("Failed to acquire next image {:?}", e),
        };
      if suboptimal {
        recreate_swapchain = true;
      }
      let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
      let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
      )
      .unwrap();
      builder
        .begin_render_pass(
          framebuffers[image_num].clone(),
          SubpassContents::Inline,
          clear_values,
        )
        .unwrap()
        .set_viewport(0, [viewport.clone()])
        .bind_pipeline_graphics(pipeline.clone())
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .draw(vertex_buffer.len() as u32, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

      let command_buffer = builder.build().unwrap();

      let future = previous_frame_end
        .take()
        .unwrap()
        .join(acquire_future)
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
        .then_signal_fence_and_flush();
      match future {
        Ok(future) => {
          previous_frame_end = Some(future.boxed());
        }
        Err(FlushError::OutOfDate) => {
          recreate_swapchain = true;
          previous_frame_end = Some(sync::now(device.clone()).boxed());
        }
        Err(e) => {
          error!("Failed to flush future: {:?}", e);
          previous_frame_end = Some(sync::now(device.clone()).boxed());
        }
      }
    }
    _ => (),
  });
}

fn window_size_dependent_setup(
  images: &[Arc<SwapchainImage<Window>>],
  render_pass: Arc<RenderPass>,
  viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
  let dimensions = images[0].dimensions().width_height();
  viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

  images
    .iter()
    .map(|image| {
      let view = ImageView::new_default(image.clone()).unwrap();
      Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
          attachments: vec![view],
          ..Default::default()
        },
      )
      .unwrap()
    })
    .collect::<Vec<_>>()
}
